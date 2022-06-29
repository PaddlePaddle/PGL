# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
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
import argparse
import time
import os
import math

import numpy as np
import paddle
import paddle.distributed.fleet as fleet
import pgl
from pgl.utils.logger import log
import yaml
from easydict import EasyDict as edict
from pgl.utils.data import Dataloader

from model import SkipGramModel
from dataset import ShardedDataset
from dataset import BatchRandWalk


def load(name):
    if name == 'cora':
        dataset = pgl.dataset.CoraDataset()
    elif name == "pubmed":
        dataset = pgl.dataset.CitationDataset("pubmed", symmetry_edges=True)
    elif name == "citeseer":
        dataset = pgl.dataset.CitationDataset("citeseer", symmetry_edges=True)
    elif name == "BlogCatalog":
        dataset = pgl.dataset.BlogCatalogDataset()
    else:
        raise ValueError(name + " dataset doesn't exists")
    indegree = dataset.graph.indegree()
    outdegree = dataset.graph.outdegree()
    return dataset.graph


def build_graph(args):
    graph = load(args.dataset)
    return graph


def train(exe, program, reader, loss, log_per_step=1):
    total_loss = 0.
    batch = 0
    reader.start()
    try:
        while True:
            num_samples = 1
            begin_time = time.time()
            loss_val, = exe.run(program, fetch_list=[loss.name])
            step_time = time.time() - begin_time
            loss_val = loss_val.mean()

            total_loss += loss_val * num_samples
            batch += 1

            if batch % log_per_step == 0:
                log.info("Batch %s\t%s-Loss %.6f\t%.6f sec/step" %
                         (batch, "train", loss_val, step_time))
    except paddle.framework.core.EOFException:
        reader.reset()

    return total_loss / batch


def StaticSkipGramModel(num_nodes,
                        neg_num,
                        embed_size,
                        sparse=False,
                        sparse_embedding=False,
                        shared_embedding=False):
    src = paddle.static.data("src", shape=[-1, 1], dtype="int64")
    dsts = paddle.static.data("dsts", shape=[-1, neg_num + 1], dtype="int64")
    py_reader = paddle.io.DataLoader.from_generator(
        capacity=64,
        feed_list=[src, dsts],
        iterable=False,
        use_double_buffer=False)
    model = SkipGramModel(
        num_nodes,
        embed_size,
        neg_num,
        sparse=sparse,
        sparse_embedding=sparse_embedding,
        shared_embedding=shared_embedding)
    loss = model(src, dsts)
    return py_reader, loss


def main(args):
    paddle.set_device("cpu")
    paddle.enable_static()

    fleet.init()

    fake_num_nodes = 1
    py_reader, loss = StaticSkipGramModel(
        fake_num_nodes,
        args.neg_num,
        args.embed_size,
        sparse_embedding=True,
        shared_embedding=args.shared_embedding)

    optimizer = paddle.optimizer.Adam(args.learning_rate, lazy_mode=True)
    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.a_sync = True
    optimizer = fleet.distributed_optimizer(optimizer, dist_strategy)
    optimizer.minimize(loss)

    # init and run server or worker
    if fleet.is_server():
        fleet.init_server()
        fleet.run_server()

    if fleet.is_worker():
        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        exe.run(paddle.static.default_startup_program())
        fleet.init_worker()

        graph = build_graph(args)
        # bind gen
        train_ds = ShardedDataset(graph.nodes, args.epoch)
        collate_fn = BatchRandWalk(graph, args.walk_len, args.win_size,
                                   args.neg_num, args.neg_sample_type)
        data_loader = Dataloader(
            train_ds,
            batch_size=args.cpu_batch_size,
            shuffle=True,
            num_workers=args.sample_workers,
            collate_fn=collate_fn)
        py_reader.set_batch_generator(lambda: data_loader)

        train_loss = train(exe,
                           paddle.static.default_main_program(), py_reader,
                           loss)
        fleet.stop_worker()

        if fleet.is_first_worker():
            fleet.save_persistables(exe, "./model",
                                    paddle.static.default_main_program())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distributed Deepwalk')
    parser.add_argument(
        "--dataset",
        type=str,
        default="BlogCatalog",
        help="dataset (cora, pubmed, BlogCatalog)")
    parser.add_argument(
        "--conf",
        type=str,
        default="./config.yaml",
        help="config file for models")
    parser.add_argument("--epoch", type=int, default=400, help="Epoch")
    args = parser.parse_args()

    # merge user args and config file 
    config = edict(yaml.load(open(args.conf), Loader=yaml.FullLoader))
    config.update(vars(args))
    main(config)
