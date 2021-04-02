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
import argparse
import time
from functools import partial

import numpy as np
import tqdm
import pgl
import paddle
from pgl.utils.logger import log
from pgl.utils.data import Dataloader

from model import GraphSage
from dataset import ShardedDataset, batch_fn
import paddle.distributed.fleet as fleet
import paddle.distributed.fleet.base.role_maker as role_maker
import paddle.fluid as F


def run(dataset,
        feature,
        exe,
        program,
        loss,
        acc,
        phase="train",
        log_per_step=5):

    batch = 0
    total_loss = 0.
    total_acc = 0.
    total_sample = 0

    for g, sample_index, index, label in dataset:
        feed_dict = {
            "num_nodes": np.array(
                [g.num_nodes], dtype="int32"),
            "edges": g.edges.astype("int32"),
            "sample_index": sample_index.astype("int32"),
            "index": index.astype("int32"),
            "label": label.astype("int64").reshape(-1),
            "feature": feature[sample_index].astype("float32")
        }

        batch += 1
        batch_loss, batch_acc = exe.run(program,
                                        feed=feed_dict,
                                        fetch_list=[loss.name, acc.name])

        if batch % log_per_step == 0:
            log.info("Batch %s %s-Loss %s %s-Acc %s" %
                     (batch, phase, batch_loss, phase, batch_acc))
        total_acc += batch_acc * len(index)
        total_loss += batch_loss * len(index)
        total_sample += len(index)

    return total_loss / total_sample, total_acc / total_sample


def build_net(input_size, num_class, hidden_size, num_layers):
    num_nodes = F.data("num_nodes", shape=[1], dtype="int32")
    edges = F.data("edges", shape=[None, 2], dtype="int32")
    sample_index = F.data("sample_index", shape=[None], dtype="int32")
    index = F.data("index", shape=[None], dtype="int32")
    label = F.data("label", shape=[None], dtype="int64")
    label = paddle.reshape(label, [-1, 1])
    graph = pgl.Graph(num_nodes=num_nodes, edges=edges)
    feat = F.data("feature", shape=[None, input_size], dtype="float32")

    model = GraphSage(
        input_size=input_size,
        num_class=num_class,
        hidden_size=hidden_size,
        num_layers=num_layers)

    g = pgl.Graph(num_nodes=num_nodes, edges=edges)
    pred = model(g, feat)
    pred = paddle.gather(pred, index)
    loss = paddle.nn.functional.cross_entropy(pred, label)
    acc = paddle.metric.accuracy(input=pred, label=label, k=1)
    return loss, acc


def main(args):
    role = role_maker.PaddleCloudRoleMaker()
    fleet.init(role)
    data = pgl.dataset.RedditDataset(args.normalize, args.symmetry)
    log.info("Preprocess finish")
    log.info("Train Examples: %s" % len(data.train_index))
    log.info("Val Examples: %s" % len(data.val_index))
    log.info("Test Examples: %s" % len(data.test_index))
    log.info("Num nodes %s" % data.graph.num_nodes)
    log.info("Num edges %s" % data.graph.num_edges)
    log.info("Average Degree %s" % np.mean(data.graph.indegree()))

    graph = data.graph
    train_index = data.train_index
    val_index = data.val_index
    test_index = data.test_index

    train_label = data.train_label
    val_label = data.val_label
    test_label = data.test_label

    loss, acc = build_net(
        input_size=data.feature.shape[-1],
        num_class=data.num_classes,
        hidden_size=args.hidden_size,
        num_layers=len(args.samples))
    test_program = paddle.static.default_main_program().clone(for_test=True)

    strategy = fleet.DistributedStrategy()
    strategy.a_sync = True
    optimizer = paddle.fluid.optimizer.Adam(learning_rate=args.lr)
    optimizer = fleet.distributed_optimizer(optimizer, strategy)
    optimizer.minimize(loss)

    if fleet.is_server():
        fleet.init_server()
        fleet.run_server()
    else:
        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        exe.run(paddle.static.default_startup_program())
        fleet.init_worker()

        train_ds = ShardedDataset(train_index, train_label)
        valid_ds = ShardedDataset(val_index, val_label)

        collate_fn = partial(batch_fn, graph=graph, samples=args.samples)

        train_loader = Dataloader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.sample_workers,
            collate_fn=collate_fn)

        valid_loader = Dataloader(
            valid_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.sample_workers,
            collate_fn=collate_fn)

        for epoch in tqdm.tqdm(range(args.epoch)):
            train_loss, train_acc = run(train_loader,
                                        data.feature,
                                        exe,
                                        paddle.static.default_main_program(),
                                        loss,
                                        acc,
                                        phase="train")

            valid_loss, valid_acc = run(valid_loader,
                                        data.feature,
                                        exe,
                                        test_program,
                                        loss,
                                        acc,
                                        phase="valid")

            log.info("Epoch %s Valid-Loss %s Valid-Acc %s" %
                     (epoch, valid_loss, valid_acc))
        test_loss, test_acc = run(test_loader,
                                  data.feature,
                                  exe,
                                  test_program,
                                  loss,
                                  acc,
                                  phase="test")
        log.info("Epoch %s Valid-Loss %s Valid-Acc %s" %
                 (epoch, test_loss, test_acc))

        fleet.stop_worker()


if __name__ == "__main__":
    paddle.enable_static()
    parser = argparse.ArgumentParser(description='graphsage')
    parser.add_argument(
        "--normalize", action='store_true', help="normalize features")
    parser.add_argument(
        "--symmetry", action='store_true', help="undirect graph")
    parser.add_argument("--sample_workers", type=int, default=5)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument('--samples', nargs='+', type=int, default=[25, 10])
    args = parser.parse_args()
    main(args)
