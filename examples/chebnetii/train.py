# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import paddle.nn as nn
from pgl.utils.logger import log
import paddle.nn.functional as F
import numpy as np
import time
import argparse
import yaml
from easydict import EasyDict as edict
import tqdm
from paddle.optimizer import Adam
from utils import load_data
from model import ChebNetII


def train(node_index, node_label, gnn_model, graph, criterion, optim):
    gnn_model.train()
    pred = gnn_model(graph, graph.node_feat["words"])
    pred = paddle.gather(pred, node_index)
    loss = criterion(pred, node_label)
    loss.backward()
    acc = paddle.metric.accuracy(input=pred, label=node_label, k=1)
    for op in optim:
        op.step()
        op.clear_grad()
    #optim.step()
    #optim.clear_grad()
    return loss, acc


@paddle.no_grad()
def eval(node_index, node_label, gnn_model, graph, criterion):
    gnn_model.eval()
    pred = gnn_model(graph, graph.node_feat["words"])
    pred = paddle.gather(pred, node_index)
    loss = criterion(pred, node_label)
    acc = paddle.metric.accuracy(input=pred, label=node_label, k=1)
    return loss, acc


def main(args):
    criterion = paddle.nn.loss.CrossEntropyLoss()
    dur = []
    best_test = []
    #10 fixed seeds for random splits.
    SEEDS = [
        1941488137, 4198936517, 983997847, 4023022221, 4019585660, 2108550661,
        1648766618, 629014539, 3212139042, 2424918363
    ]

    for run in tqdm.tqdm(range(args.runs)):
        args.seed = SEEDS[run]
        dataset = load_data(args.dataset, args.seed)

        graph = dataset.graph
        train_index = dataset.train_index
        train_label = dataset.train_label

        val_index = dataset.val_index
        val_label = dataset.val_label

        test_index = dataset.test_index
        test_label = dataset.test_label

        cal_val_acc = []
        cal_test_acc = []
        cal_val_loss = []
        cal_test_loss = []
        gnn_model = ChebNetII(
            input_size=graph.node_feat["words"].shape[1],
            num_class=dataset.num_classes,
            hidden_size=args.hidden,
            K=args.K,
            drop=args.dropout,
            dprate=args.dprate)
        optim = []
        optim1 = Adam(
            learning_rate=args.lr,
            parameters=gnn_model.linear_1.parameters(),
            weight_decay=args.weight_decay)
        optim2 = Adam(
            learning_rate=args.lr,
            parameters=gnn_model.linear_2.parameters(),
            weight_decay=args.weight_decay)
        optim3 = Adam(
            learning_rate=args.prop_lr,
            parameters=gnn_model.prop.parameters(),
            weight_decay=args.prop_wd)
        optim.append(optim1)
        optim.append(optim2)
        optim.append(optim3)

        best_val_acc = test_acc = 0
        best_val_loss = float('inf')
        val_loss_history = []
        val_acc_history = []
        for epoch in range(args.epoch):
            if epoch >= 3:
                start = time.time()
            train_loss, train_acc = train(train_index, train_label, gnn_model,
                                          graph, criterion, optim)
            if epoch >= 3:
                end = time.time()
                dur.append(end - start)
            val_loss, val_acc = eval(val_index, val_label, gnn_model, graph,
                                     criterion)
            test_loss, tmp_test_acc = eval(test_index, test_label, gnn_model,
                                           graph, criterion)

            if val_loss < best_val_loss:
                best_val_acc = val_acc
                best_val_loss = val_loss
                test_acc = tmp_test_acc
                TEST = gnn_model.prop.temp.clone()
                theta = F.relu(TEST.detach().cpu()).numpy()

            if epoch >= 0:
                val_loss_history.append(val_loss.numpy())
                val_acc_history.append(val_acc.numpy())

                if args.early_stopping > 0 and epoch > args.early_stopping:
                    tmp = val_loss_history[-(args.early_stopping + 1):-1]
                    if val_loss > np.mean(tmp):
                        #print('The sum of epochs:',epoch)
                        break
        #if theta is not None:
        #    print('The coe of '+args.net+' is:', theta)
        log.info("Runs %s: Model: Best Test Accuracy: %f" % (run, test_acc))
        best_test.append(test_acc)
    log.info("Average Speed %s ms/ epoch" % (np.mean(dur) * 1000))
    log.info("Dataset: %s Best Test Accuracy: %f ± %f " %
             (args.dataset, np.mean(best_test) * 100, np.std(best_test) * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChebNetII')
    parser.add_argument('--seed', type=int, default=42, help='seed.')
    parser.add_argument(
        "--dataset", type=str, default="cora", help="dataset (cora, pubmed)")
    parser.add_argument("--epoch", type=int, default=1000, help="Epoch")
    parser.add_argument(
        "--runs", type=int, default=10, help="runs, fixed 10 random splits")
    parser.add_argument(
        "--net", type=str, default='ChebNetII', help="the model")

    parser.add_argument(
        '--lr', type=float, default=0.01, help='learning rate.')
    parser.add_argument(
        '--weight_decay', type=float, default=0.0005, help='weight decay.')
    parser.add_argument(
        '--early_stopping', type=int, default=200, help='early stopping.')
    parser.add_argument('--hidden', type=int, default=64, help='hidden units.')
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.5,
        help='dropout for neural networks.')
    parser.add_argument('--K', type=int, default=10, help='propagation steps.')

    parser.add_argument(
        '--dprate',
        type=float,
        default=0.5,
        help='dropout for propagation layer.')
    parser.add_argument(
        '--prop_lr',
        type=float,
        default=0.01,
        help='learning rate for propagation layer.')
    parser.add_argument(
        '--prop_wd',
        type=float,
        default=0.0005,
        help='learning rate for propagation layer.')

    args = parser.parse_known_args()[0]
    print(args)
    main(args)
