# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
    Train GCN model on Citation Network(Cora/Pubmed/Citeseer).
"""

import sys
import argparse
import numpy as np
from functools import partial

import paddle
import pgl
from pgl.utils.logger import log
from pgl.utils.data.dataloader import Dataloader

sys.path.append("..")
from gnn_models import GCN
from dataloader import PartitionDataset, EvalPartitionDataset
from dataloader import subdata_batch_fn
from partition import metis_graph_partition
from utils import check_device
from utils import process_batch_data, compute_acc, gen_mask, permute


def load(data_name):
    data_name = data_name.lower()
    if data_name == 'cora':
        data = pgl.dataset.CoraDataset()
    elif data_name == 'pubmed':
        data = pgl.dataset.CitationDataset("pubmed", symmetry_edges=True)
    elif data_name == 'citeseer':
        data = pgl.dataset.CitationDataset("citeseer", symmetry_edges=True)
    else:
        raise ValueError(name + " dataset doesn't exists")

    indegree = data.graph.indegree()
    data.graph.node_feat["words"] = normalize(data.graph.node_feat["words"])

    data.feature = data.graph.node_feat["words"]
    data.train_mask = gen_mask(data.graph.num_nodes, data.train_index)
    data.val_mask = gen_mask(data.graph.num_nodes, data.val_index)
    data.test_mask = gen_mask(data.graph.num_nodes, data.test_index)
    return data


def normalize(feat):
    return feat / np.maximum(np.sum(feat, -1, keepdims=True), 1)


def train(dataloader, model, feature, norm, label, train_mask, criterion,
          optim, epoch, gen_train_data_in_advance):
    model.train()

    batch_id = 0
    total_loss = total_examples = 0

    if gen_train_data_in_advance:
        np.random.shuffle(dataloader)

    for batch_data in dataloader:
        batch_id += 1
        g, batch_size, n_id, offset, count, feat, sub_norm = \
            process_batch_data(batch_data, feature, norm)
        pred = model(g, feat, sub_norm, batch_size, n_id, offset, count)
        pred = pred[:batch_size]

        sub_train_mask = paddle.gather(train_mask, n_id[:batch_size])
        y = paddle.gather(label, n_id[:batch_size])
        true_index = paddle.nonzero(sub_train_mask)
        if true_index.shape[0] == 0:
            continue
        pred = paddle.gather(pred, true_index)
        y = paddle.gather(y, true_index)

        loss = criterion(pred, y)
        loss.backward()
        optim.step()
        optim.clear_grad()

        total_loss += loss.numpy() * true_index.shape[0]
        total_examples += true_index.shape[0]

    return total_loss / total_examples


@paddle.no_grad()
def eval(graph, model, feature, norm, label, mask):
    # Since the graph is small, we can get full-batch inference.
    model.eval()
    feature = paddle.to_tensor(feature)
    norm = paddle.to_tensor(norm)
    out = model(graph, feature, norm)
    acc = compute_acc(out, label, mask)
    return acc


def main(args):
    log.info("Loading data...")
    data = load(args.dataset)
    num_classes = data.num_classes

    log.info("Running into %d metis partitions..." % args.num_parts)
    permutation, split = metis_graph_partition(
        data.graph, npart=args.num_parts)

    log.info("Permuting data...")
    data, feature = permute(data, data.feature, permutation)
    graph = data.graph

    log.info("Building data loader for training and validation...")
    dataset = PartitionDataset(split)
    collate_fn = partial(
        subdata_batch_fn,
        graph=graph,
        split=split,
        flag_buffer=np.zeros(
            graph.num_nodes, dtype="int32"))
    train_loader = Dataloader(
        dataset,
        batch_size=args.batch_size,
        drop_last=False,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn)
    if args.gen_train_data_in_advance:
        train_loader = list(train_loader)
    if args.gcn_norm:
        degree = graph.indegree()
        gcn_norm = degree.astype(np.float32)
        gcn_norm = np.clip(gcn_norm, 1.0, np.max(gcn_norm))
        gcn_norm = np.power(gcn_norm, -0.5)
        gcn_norm = np.reshape(gcn_norm, [-1, 1])
    else:
        gcn_norm = None

    # If you want to know how to calculate buffer_size,
    # you can just try diffrent numbers until the program works normally,
    # or you can turn to run_gcn_reddit.py to see how to calculate.
    if args.dataset == 'cora':
        buffer_size = 500
    elif args.dataset == 'pubmed':
        buffer_size = 2000
    else:
        buffer_size = 1000

    model = GCN(
        num_nodes=graph.num_nodes,
        num_layers=args.num_layers,
        input_size=feature.shape[1],
        hidden_size=args.hidden_size,
        output_size=num_classes,
        dropout=0.,
        pool_size=args.num_layers - 1,
        buffer_size=buffer_size, )

    optim = paddle.optimizer.Adam(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        parameters=model.parameters())

    criterion = paddle.nn.loss.CrossEntropyLoss()

    eval_graph = pgl.Graph(edges=graph.edges, num_nodes=graph.num_nodes)
    eval_graph.tensor()
    best_valid_acc = final_test_acc = 0
    for epoch in range(args.epoch):
        loss = train(train_loader, model, feature, gcn_norm, data.label,
                     data.train_mask, criterion, optim, epoch,
                     args.gen_train_data_in_advance)
        train_acc = eval(eval_graph, model, feature, gcn_norm, data.label,
                         data.train_mask)
        valid_acc = eval(eval_graph, model, feature, gcn_norm, data.label,
                         data.val_mask)
        test_acc = eval(eval_graph, model, feature, gcn_norm, data.label,
                        data.test_mask)
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            final_test_acc = test_acc
        log.info(
            f"Epoch:%d, Train Loss: %.4f, Train Acc: %.4f, Valid Acc: %.4f "
            f"Test Acc: %.4f, Final Acc: %.4f" % (
                epoch, loss, train_acc, valid_acc, test_acc, final_test_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training GCN on Citation Network')
    parser.add_argument(
        "--dataset", type=str, default="cora", help="Cora, Pubmed, Citeseer")
    parser.add_argument(
        "--epoch", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument(
        "--num_parts", type=int, default=40, help="Number of graph partitions")
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of gcn layers")
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=16,
        help="Hidden size in gcn models")
    parser.add_argument(
        "--weight_decay", type=float, default=5e-4, help="Weight decay rate")
    parser.add_argument(
        "--gcn_norm",
        action="store_true",
        help="Whether generate gcn norm feature")
    parser.add_argument(
        "--batch_size", type=int, default=10, help="Batch size")
    parser.add_argument(
        "--gen_train_data_in_advance",
        action="store_true",
        help="Whether generate train batch data in advance")
    args = parser.parse_args()

    log.info("Checking device...")
    if not check_device():
        log.info(
            f"Current device does not meet GNNAutoScale running conditions. "
            f"We should run GNNAutoScale under GPU and CPU environment simultaneously."
            f"This program will exit.")
    else:
        log.info(args)
        main(args)
