# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
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
import pgl
import model
import paddle
import paddle.nn as nn
from pgl.utils.logger import log
import numpy as np
import time
import argparse
import yaml
from easydict import EasyDict as edict
import tqdm
from paddle.optimizer import Adam
import paddle.distributed as dist


def normalize(feat):
    return feat / np.maximum(np.sum(feat, -1, keepdims=True), 1)


def load(name, normalized_feature=True):
    if name == 'cora':
        dataset = pgl.dataset.CoraDataset()
    elif name == "pubmed":
        dataset = pgl.dataset.CitationDataset("pubmed", symmetry_edges=True)
    elif name == "citeseer":
        dataset = pgl.dataset.CitationDataset("citeseer", symmetry_edges=True)
    else:
        raise ValueError(name + " dataset doesn't exists")

    indegree = dataset.graph.indegree()
    dataset.graph.node_feat["words"] = normalize(dataset.graph.node_feat[
        "words"])

    if paddle.distributed.get_world_size() > 1:
        dataset.graph = pgl.DistGPUGraph(dataset.graph)
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


def train(node_index, node_label, gnn_model, graph, criterion, optim):
    gnn_model.train()
    pred = gnn_model(graph, graph.node_feat["words"])
    pred = paddle.gather(pred, node_index)
    loss = criterion(pred, node_label)
    loss.backward()
    acc = paddle.metric.accuracy(input=pred, label=node_label, k=1)
    optim.step()
    optim.clear_grad()
    return loss, acc


@paddle.no_grad()
def eval(node_index, node_label, gnn_model, graph, criterion):
    gnn_model.eval()
    pred = gnn_model(graph, graph.node_feat["words"])
    pred = paddle.gather(pred, node_index)
    loss = criterion(pred, node_label)
    acc = paddle.metric.accuracy(input=pred, label=node_label, k=1)
    return loss, acc


def set_seed(seed):
    paddle.seed(seed)
    np.random.seed(seed)


def main(args, config, run):
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    dataset = load(args.dataset, args.feature_pre_normalize)
    graph = dataset.graph

    train_index = dataset.train_index
    train_label = dataset.train_label

    val_index = dataset.val_index
    val_label = dataset.val_label

    test_index = dataset.test_index
    test_label = dataset.test_label

    GraphModel = getattr(model, config.model_name)
    criterion = paddle.nn.loss.CrossEntropyLoss()

    dur = []

    best_test = []

    cal_val_acc = []
    cal_test_acc = []
    cal_val_loss = []
    cal_test_loss = []

    gnn_model = GraphModel(
        input_size=graph.node_feat["words"].shape[1],
        num_class=dataset.num_classes,
        **config)

    if paddle.distributed.get_world_size() > 1:
        gnn_model = paddle.DataParallel(gnn_model)

    optim = Adam(
        learning_rate=config.learning_rate,
        parameters=gnn_model.parameters(),
        weight_decay=config.weight_decay)

    for epoch in tqdm.tqdm(range(args.epoch)):
        train_loss, train_acc = train(train_index, train_label, gnn_model,
                                      graph, criterion, optim)
        val_loss, val_acc = eval(val_index, val_label, gnn_model, graph,
                                 criterion)
        cal_val_acc.append(val_acc.numpy())
        cal_val_loss.append(val_loss.numpy())

        test_loss, test_acc = eval(test_index, test_label, gnn_model, graph,
                                   criterion)
        cal_test_acc.append(test_acc.numpy())
        cal_test_loss.append(test_loss.numpy())

    log.info("Runs %s: Model: %s Best Test Accuracy: %f" %
             (run, config.model_name, cal_test_acc[np.argmin(cal_val_loss)]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmarking Citation Network')
    parser.add_argument(
        "--dataset", type=str, default="cora", help="dataset (cora, pubmed)")
    parser.add_argument("--conf", type=str, help="config file for models")
    parser.add_argument("--epoch", type=int, default=200, help="Epoch")
    parser.add_argument("--runs", type=int, default=10, help="runs")
    parser.add_argument(
        "--feature_pre_normalize",
        type=bool,
        default=True,
        help="pre_normalize feature")
    args = parser.parse_args()
    config = edict(yaml.load(open(args.conf), Loader=yaml.FullLoader))
    log.info(args)
    for run in range(args.runs):
        dist.spawn(main, args=(args, config, run), nprocs=-1)
