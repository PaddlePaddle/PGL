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
    Train GCN model on Reddit dataset.
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
from utils import check_device, time_wrapper
from utils import process_batch_data, compute_acc, gen_mask, permute


def load(args):
    data = pgl.dataset.RedditDataset(args.normalize, args.symmetry)
    y = np.zeros(data.graph.num_nodes, dtype="int64")
    y[data.train_index] = data.train_label
    y[data.val_index] = data.val_label
    y[data.test_index] = data.test_label
    data.y = y
    data.train_mask = gen_mask(data.graph.num_nodes, data.train_index)
    data.val_mask = gen_mask(data.graph.num_nodes, data.val_index)
    data.test_mask = gen_mask(data.graph.num_nodes, data.test_index)

    return data


def train(dataloader, model, feature, gcn_norm, label, train_mask, criterion,
          optim, epoch, gen_train_data_in_advance):
    model.train()

    batch_id = 0
    total_loss = total_examples = 0

    if gen_train_data_in_advance:
        np.random.shuffle(dataloader)

    for batch_data in dataloader:
        batch_id += 1
        g, batch_size, n_id, offset, count, feat, sub_norm = \
            process_batch_data(batch_data, feature, gcn_norm)
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


@time_wrapper("Eval")
@paddle.no_grad()
def eval(graph, loader, model, feature, norm):
    model.eval()
    return model(subgraph=graph, x=feature, norm=norm, loader=loader)


def main(args):
    log.info("Loading data...")
    data = load(args)

    log.info("Running into %d metis partitions..." % args.num_parts)
    permutation, part = metis_graph_partition(data.graph, npart=args.num_parts)

    log.info("Permuting data...")
    data, feature = permute(data, data.feature, permutation)
    graph = data.graph

    log.info("Building data loader for training and validation...")
    dataset = PartitionDataset(part)
    collate_fn = partial(
        subdata_batch_fn,
        graph=graph,
        part=part,
        flag_buffer=np.zeros(
            graph.num_nodes, dtype="int32"))
    train_loader = Dataloader(
        dataset,
        batch_size=args.batch_size,
        drop_last=False,
        shuffle=True,
        num_workers=5,
        collate_fn=collate_fn)
    if args.gen_train_data_in_advance:
        train_loader = list(train_loader)
    eval_dataset = EvalPartitionDataset(
        graph,
        part,
        args.batch_size,
        flag_buffer=np.zeros(
            graph.num_nodes, dtype="int32"))
    eval_loader = eval_dataset.data_list

    if args.gcn_norm:
        degree = graph.indegree()
        gcn_norm = degree.astype(np.float32)
        gcn_norm = np.clip(gcn_norm, 1.0, np.max(gcn_norm))
        gcn_norm = np.power(gcn_norm, -0.5)
        gcn_norm = np.reshape(gcn_norm, [-1, 1])
    else:
        gcn_norm = None

    log.info("Calculating buffer size...")
    buffer_size = -1
    for subdata in eval_loader:
        _, _, n_id, _, _, _, _ = process_batch_data(subdata)
        buffer_size = max(len(n_id), buffer_size)
    log.info("Buffer size: %d" % buffer_size)

    model = GCN(
        num_nodes=graph.num_nodes,
        num_layers=args.num_layers,
        input_size=feature.shape[1],
        hidden_size=args.hidden_size,
        output_size=data.num_classes,
        dropout=0.5,
        pool_size=args.num_layers - 1,
        buffer_size=buffer_size, )

    optim = paddle.optimizer.Adam(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        parameters=model.parameters())

    criterion = paddle.nn.loss.CrossEntropyLoss()

    best_val_acc = final_test_acc = 0
    for epoch in range(args.epoch):
        loss = train(train_loader, model, feature, gcn_norm, data.label,
                     data.train_mask, criterion, optim, epoch,
                     args.gen_train_data_in_advance)
        out = eval(graph, eval_loader, model, feature, gcn_norm)
        val_acc = compute_acc(out.cuda(), data.label, data.val_mask)
        test_acc = compute_acc(out.cuda(), data.label, data.test_mask)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            final_test_acc = test_acc
        log.info(f"Epoch:%d, Train Loss: %.4f, Valid Acc: %.4f "
                 f"Test Acc: %.4f, Final Acc: %.4f" %
                 (epoch, loss, val_acc, test_acc, final_test_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training GCN on Reddit dataset')
    parser.add_argument(
        "--normalize",
        default=True,
        action="store_false",
        help="normalize features")
    parser.add_argument(
        "--symmetry",
        default=True,
        action="store_false",
        help="undirect graph")
    parser.add_argument(
        "--epoch", type=int, default=400, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument(
        "--num_parts",
        type=int,
        default=200,
        help="Number of graph partitions")
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of gcn layers")
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=256,
        help="Hidden size in gcn models")
    parser.add_argument(
        "--weight_decay", type=float, default=5e-4, help="Weight decay rate")
    parser.add_argument(
        "--gcn_norm",
        action="store_true",
        help="Whether generate gcn norm feature")
    parser.add_argument(
        "--batch_size", type=int, default=100, help="Batch size")
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
