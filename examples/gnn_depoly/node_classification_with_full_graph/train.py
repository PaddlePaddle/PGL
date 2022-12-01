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
from pgl.utils.logger import log
from paddle.optimizer import Adam

sys.path.insert(0, os.path.abspath(".."))
from models import GCN, GAT, GraphSage


def normalize(feat):
    return feat / np.maximum(np.sum(feat, -1, keepdims=True), 1)


def load():
    dataset = pgl.dataset.CoraDataset()

    indegree = dataset.graph.indegree()
    dataset.graph.node_feat["words"] = normalize(dataset.graph.node_feat[
        "words"])

    dataset.graph.tensor()
    train_index = dataset.train_index
    dataset.train_label = paddle.to_tensor(
        np.expand_dims(dataset.y[train_index], -1))
    dataset.train_index = paddle.to_tensor(np.expand_dims(train_index, -1))

    val_index = dataset.val_index
    dataset.val_label = paddle.to_tensor(
        np.expand_dims(dataset.y[val_index], -1))
    dataset.val_index = paddle.to_tensor(np.expand_dims(val_index, -1))

    test_index = dataset.test_index
    dataset.test_label = paddle.to_tensor(
        np.expand_dims(dataset.y[test_index], -1))
    dataset.test_index = paddle.to_tensor(np.expand_dims(test_index, -1))

    return dataset


def train(node_index, node_label, gnn_model, graph, criterion, optim, args):
    gnn_model.train()
    if args.model == "GraphSage":
        pred = gnn_model(graph.edges, graph.node_feat["words"])
    else:
        pred = gnn_model(graph.edges, graph.num_nodes,
                         graph.node_feat["words"])
    pred = paddle.gather(pred, node_index)
    loss = criterion(pred, node_label)
    loss.backward()
    acc = paddle.metric.accuracy(input=pred, label=node_label, k=1)
    optim.step()
    optim.clear_grad()
    return loss, acc


@paddle.no_grad()
def eval(node_index, node_label, gnn_model, graph, criterion, args):
    gnn_model.eval()
    if args.model == "GraphSage":
        pred = gnn_model(graph.edges, graph.node_feat["words"])
    else:
        pred = gnn_model(graph.edges, graph.num_nodes,
                         graph.node_feat["words"])
    pred = paddle.gather(pred, node_index)
    loss = criterion(pred, node_label)
    acc = paddle.metric.accuracy(input=pred, label=node_label, k=1)
    return loss, acc


def main(args):
    dataset = load()

    graph = dataset.graph
    train_index = dataset.train_index
    train_label = dataset.train_label

    val_index = dataset.val_index
    val_label = dataset.val_label
    # We will use test dataset to deploy and infer.

    criterion = nn.loss.CrossEntropyLoss()

    if args.model == "GAT":
        gnn_model = GAT(input_size=graph.node_feat["words"].shape[1],
                        num_class=dataset.num_classes,
                        num_layers=2,
                        feat_drop=0.6,
                        attn_drop=0.6,
                        num_heads=8,
                        hidden_size=8)
    elif args.model == "GCN":
        gnn_model = GCN(input_size=graph.node_feat["words"].shape[1],
                        num_class=dataset.num_classes,
                        num_layers=1,
                        dropout=0.5,
                        hidden_size=16)
    elif args.model == "GraphSage":
        gnn_model = GraphSage(
            input_size=graph.node_feat["words"].shape[1],
            num_class=dataset.num_classes,
            num_layers=2,
            hidden_size=16)
    else:
        raise ValueError("%s model is supported!" % args.model)

    optim = Adam(
        learning_rate=0.005,
        parameters=gnn_model.parameters(),
        weight_decay=0.0005)

    best_val_acc = 0
    for epoch in range(args.epoch):
        train_loss, train_acc = train(train_index, train_label, gnn_model,
                                      graph, criterion, optim, args)
        val_loss, val_acc = eval(val_index, val_label, gnn_model, graph,
                                 criterion, args)
        if val_acc > best_val_acc:
            paddle.save(gnn_model.state_dict(),
                        "%s.pdparam" % args.model.lower())
            best_val_acc = val_acc

    log.info("Model: %s, Best Val Accuracy: %f" % (args.model, best_val_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Node Classification, no need sampling.')
    parser.add_argument(
        "--model", type=str, default="GCN", help="GCN, GAT, GraphSage")
    parser.add_argument("--epoch", type=int, default=200, help="Epoch")
    parser.add_argument(
        "--feature_pre_normalize",
        type=bool,
        default=True,
        help="pre_normalize feature")
    args = parser.parse_args()
    log.info(args)
    main(args)
