# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import glob
import numpy as np
import pandas as pd
import argparse
import time

import paddle
import pgl
from pgl.utils.logger import log

from model import RGCN


def build_heter_graph(data_path, num_nodes):
    edges = {}
    idx = 0
    for filename in glob.glob(os.path.join(data_path, '*')):
        try:
            e = pd.read_csv(filename, header=None, sep="\t").values
            edges['etype%s' % idx] = e
            idx += 1
        except Exception as e:
            log.info(e)
            continue

    node_types = [(i, "n") for i in range(num_nodes)]

    hg = pgl.HeterGraph(edges=edges, node_types=node_types)

    return hg


def main(args):
    paddle.seed(args.seed)
    np.random.seed(args.seed)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    node_labels = pd.read_csv(
        os.path.join(args.data_path, 'node_labels.txt'), header=None,
        sep="\t").values.astype("int64")
    node_labels = node_labels[:, 1:2]
    print(node_labels.shape)
    num_nodes = len(node_labels)
    train_idx = pd.read_csv(
        os.path.join(args.data_path, 'train_idx.txt'), header=None,
        sep="\t").values.astype("int64").reshape(-1, ).tolist()
    test_idx = pd.read_csv(
        os.path.join(args.data_path, 'test_idx.txt'), header=None,
        sep="\t").values.astype("int64").reshape(-1, ).tolist()

    g = build_heter_graph(os.path.join(args.data_path, "edges"), num_nodes)

    model = RGCN(
        num_nodes=num_nodes,
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_class=args.num_class,
        num_layers=args.num_layers,
        etypes=g.edge_types,
        num_bases=args.num_bases, )

    model = paddle.DataParallel(model)

    criterion = paddle.nn.loss.CrossEntropyLoss()

    optim = paddle.optimizer.Adam(
        learning_rate=args.lr,
        parameters=model.parameters(),
        weight_decay=0.001)

    test_acc_list = []
    g.tensor()
    train_labels = paddle.to_tensor(node_labels[train_idx])
    test_labels = paddle.to_tensor(node_labels[test_idx])
    train_idx = paddle.to_tensor(train_idx)
    test_idx = paddle.to_tensor(test_idx)
    for epoch in range(args.epochs):
        logits = model(g)
        train_logits = paddle.gather(logits, train_idx)
        train_loss = criterion(train_logits, train_labels)
        train_loss.backward()
        train_acc = paddle.metric.accuracy(
            train_logits, label=train_labels, k=1)
        optim.step()
        optim.clear_grad()

        test_logits = paddle.gather(logits, test_idx)
        test_loss = criterion(test_logits, test_labels)
        test_acc = paddle.metric.accuracy(test_logits, label=test_labels, k=1)

        msg = "epoch: %s" % epoch
        msg += " | train_loss: %.4f | train_acc: %.4f" \
                % (float(train_loss), float(train_acc))

        msg += " | test_loss: %.4f | test_acc: %.4f" \
                % (float(test_loss), float(test_acc))

        log.info(msg)
        test_acc_list.append(float(test_acc))

    log.info("best test acc result: %.4f" % (np.max(test_acc_list)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='graphsage')
    parser.add_argument("--data_path", type=str, default="./mutag_data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--input_size", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_class", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_bases", type=int, default=8)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()
    log.info(args)
    main(args)
