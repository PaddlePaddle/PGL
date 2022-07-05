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
from dataset import BatchNode2vecWalk


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


def train(exe, program, data_loader, loss, log_per_step=1):
    total_loss = 0.
    total_sample = 0
    for batch, (src, dsts) in enumerate(data_loader):
        num_samples = len(src)
        feed_dict = {"src": src, "dsts": dsts}
        begin_time = time.time()
        loss_val, = exe.run(program, fetch_list=[loss.name], feed=feed_dict)
        step_time = time.time() - begin_time
        loss_val = loss_val.mean()

        total_loss += loss_val * num_samples
        total_sample += num_samples

        if batch % log_per_step == 0:
            log.info("Batch %s\t%s-Loss %.6f\t%.6f sec/step" %
                     (batch, "train", loss_val, step_time))

    return total_loss / total_sample


def StaticSkipGramModel(num_nodes, neg_num, embed_size, sparse):
    src = paddle.static.data("src", shape=[-1, 1], dtype="int64")
    dsts = paddle.static.data("dsts", shape=[-1, neg_num + 1], dtype="int64")
    model = SkipGramModel(num_nodes, embed_size, neg_num, sparse)
    loss = model(src, dsts)
    return loss


def main(args):
    paddle.set_device("cpu")
    paddle.enable_static()

    fleet.init()

    if args.num_nodes is None:
        num_nodes = load(args.dataset).num_nodes
    else:
        num_nodes = args.num_nodes

    loss = StaticSkipGramModel(
        num_nodes, args.neg_num, args.embed_size, sparse=True)

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

        graph = load(args.dataset)
        # bind gen
        train_ds = ShardedDataset(graph.nodes)
        collate_fn = BatchNode2vecWalk(graph, args.walk_len, args.win_size,
                                       args.neg_num, args.neg_sample_type,
                                       args.p, args.q)
        data_loader = Dataloader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.sample_workers,
            collate_fn=collate_fn)

        cpu_num = int(os.environ.get('CPU_NUM', 1))
        if int(cpu_num) > 1:
            parallel_places = [paddle.CPUPlace()] * cpu_num
            exec_strategy = paddle.static.ExecutionStrategy()
            exec_strategy.num_threads = int(cpu_num)
            build_strategy = paddle.static.BuildStrategy()
            build_strategy.reduce_strategy = paddle.static.BuildStrategy.ReduceStrategy.Reduce
            compiled_prog = paddle.static.CompiledProgram(
                paddle.static.default_main_program()).with_data_parallel(
                    loss_name=loss.name,
                    places=parallel_places,
                    build_strategy=build_strategy,
                    exec_strategy=exec_strategy)
        else:
            compiled_prog = paddle.static.default_main_program()

        for epoch in range(args.epoch):
            train_loss = train(exe, compiled_prog, data_loader, loss)
            log.info("Runing epoch:%s\t train_loss:%.6f", epoch, train_loss)
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
    parser.add_argument("--num_nodes", type=int, default=None, help="Epoch")
    args = parser.parse_args()

    # merge user args and config file 
    config = edict(yaml.load(open(args.conf), Loader=yaml.FullLoader))
    config.update(vars(args))
    main(config)
