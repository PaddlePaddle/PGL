# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
import argparse

import numpy as np
import paddle
import pgl
import paddle.nn as nn
from paddle.io import Dataset, DataLoader
from pgl.utils.logger import log
from paddle.optimizer import Adam
from paddle.framework import core

from dataset import ShardedDataset
sys.path.insert(0, os.path.abspath(".."))
from models import GraphSage, GAT


def load_reddit_data():
    dataset = pgl.dataset.RedditDataset()
    log.info("Preprocess finish")
    log.info("Train Examples: %s" % len(dataset.train_index))
    log.info("Val Examples: %s" % len(dataset.val_index))
    log.info("Test Examples: %s" % len(dataset.test_index))
    log.info("Num nodes %s" % dataset.graph.num_nodes)
    log.info("Num edges %s" % dataset.graph.num_edges)
    log.info("Average Degree %s" % np.mean(dataset.graph.indegree()))

    graph = dataset.graph
    train_index = dataset.train_index
    val_index = dataset.val_index
    test_index = dataset.test_index

    train_label = dataset.train_label
    val_label = dataset.val_label
    test_label = dataset.test_label
    feature = dataset.feature

    return graph, (train_index, val_index, test_index), \
           (train_label, val_label, test_label), feature


def get_basic_graph_sample_neighbors_info(graph, mode="uva"):
    u = graph.edges[:, 0]
    v = graph.edges[:, 1]
    _, row, _, _, colptr = pgl.graph_kernel.build_index(v, u, graph.num_nodes)
    row = row.astype(np.int64)

    if mode == "uva":
        row = core.to_uva_tensor(row)
    else:
        row = paddle.to_tensor(row)

    colptr = paddle.to_tensor(colptr, dtype="int64")
    return row, colptr


def get_sample_graph_list(row, colptr, nodes, sample_sizes):
    graph_list = []
    for size in sample_sizes:
        neighbors, neighbor_counts = paddle.geometric.sample_neighbors(
            row, colptr, nodes, sample_size=size)
        edge_src, edge_dst, sample_index = paddle.geometric.reindex_graph(
            nodes, neighbors, neighbor_counts)
        nodes = sample_index
    edge_index = paddle.concat(
        [edge_src.reshape([-1, 1]), edge_dst.reshape([-1, 1])], axis=1)
    return edge_index, nodes


def train(data_loader, samples, row, colptr, model, feature, criterion, optim,
          args):
    model.train()
    total_loss = total_acc = 0
    total_sample = 0
    batch = 0
    for node_index, node_label in data_loader:
        batch += 1
        edge_index, n_id = get_sample_graph_list(row, colptr, node_index,
                                                 samples)

        if args.model == "GraphSage":
            pred = model(edge_index, feature[n_id])[:node_index.shape[0]]
        elif args.model == "GAT":
            pred = model(edge_index, n_id.shape[0],
                         feature[n_id])[:node_index.shape[0]]
        else:
            raise ValueError("Not supported model!")
        loss = criterion(pred, node_label)
        acc = paddle.metric.accuracy(
            input=pred, label=node_label.reshape([-1, 1]), k=1)
        total_acc += acc.numpy() * node_index.shape[0]
        loss.backward()
        optim.step()
        optim.clear_grad()
        total_loss += float(loss) * node_index.shape[0]
        total_sample += node_index.shape[0]

    return total_loss / total_sample, total_acc / total_sample


@paddle.no_grad()
def eval(data_loader, samples, row, colptr, model, feature, criterion, args):
    model.eval()
    total_acc = total_loss = 0
    total_sample = 0
    for node_index, node_label in data_loader:
        batch_size = node_index.shape[0]
        edge_index, n_id = get_sample_graph_list(row, colptr, node_index,
                                                 samples)

        if args.model == "GraphSage":
            pred = model(edge_index, feature[n_id])[:node_index.shape[0]]
        elif args.model == "GAT":
            pred = model(edge_index, n_id.shape[0],
                         feature[n_id])[:node_index.shape[0]]
        else:
            raise ValueError("Not supported model!")
        loss = criterion(pred, node_label)
        acc = paddle.metric.accuracy(
            input=pred, label=node_label.reshape([-1, 1]), k=1)
        total_acc += acc.numpy() * batch_size
        total_loss += float(loss)
        total_sample += batch_size
    return total_loss / len(data_loader), total_acc / total_sample


def main(args):
    graph, (train_index, val_index, _), (train_label, val_label, _), \
        feature = load_reddit_data()
    row, colptr = get_basic_graph_sample_neighbors_info(graph, args.mode)
    train_ds = ShardedDataset(train_index, train_label)
    val_ds = ShardedDataset(val_index, val_label)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False)

    if args.model == "GraphSage":
        gnn_model = GraphSage(feature.shape[1], 41,
                              len(args.samples), args.hidden_size)
    elif args.model == "GAT":
        gnn_model = GAT(feature.shape[1],
                        41,
                        num_layers=2,
                        hidden_size=args.hidden_size)

    criterion = paddle.nn.loss.CrossEntropyLoss()
    optim = paddle.optimizer.Adam(
        learning_rate=args.lr,
        parameters=gnn_model.parameters(),
        weight_decay=args.weight_decay)
    feature = paddle.to_tensor(feature)

    best_val_acc = 0
    for i in range(args.epochs):
        train_loss, train_acc = train(train_loader, args.samples, row, colptr,
                                      gnn_model, feature, criterion, optim,
                                      args)
        val_loss, val_acc = eval(val_loader, args.samples, row, colptr,
                                 gnn_model, feature, criterion, args)
        if best_val_acc < val_acc:
            paddle.save(gnn_model.state_dict(),
                        "%s.pdparam" % args.model.lower())
            best_val_acc = val_acc
        log.info("Epoch :%s, Train loss: %s, Train acc: %s, Val acc: %s" %
                 (i, train_loss, train_acc, val_acc))
    log.info("Best val acc: %s" % best_val_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PGL')
    parser.add_argument("--model", type=str, default="GraphSage")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--weight_decay", type=float, default=0., help="Weight for L2 loss")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument('--samples', nargs='+', type=int, default=[25, 10])
    parser.add_argument("--mode", type=str, default="uva", help="uva, gpu")
    args = parser.parse_args()
    print(args)
    main(args)
