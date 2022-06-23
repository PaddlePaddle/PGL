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


def StaticSkipGramModel(num_nodes,
                        neg_num,
                        embed_size,
                        num_emb_part=8,
                        shared_embedding=False):
    src = paddle.static.data("src", shape=[-1, 1], dtype="int64")
    dsts = paddle.static.data("dsts", shape=[-1, neg_num + 1], dtype="int64")
    model = SkipGramModel(
        num_nodes,
        embed_size,
        neg_num,
        num_emb_part,
        shared_embedding=shared_embedding)
    loss = model(src, dsts)
    return loss


def main(args):
    paddle.enable_static()
    paddle.set_device('gpu:%d' % paddle.distributed.ParallelEnv().dev_id)

    fleet.init(is_collective=True)

    graph = load(args.dataset)

    loss = StaticSkipGramModel(
        graph.num_nodes,
        args.neg_num,
        args.embed_size,
        num_emb_part=args.num_emb_part,
        shared_embedding=args.shared_embedding)

    optimizer = paddle.optimizer.Adam(args.learning_rate)
    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.sharding = True
    dist_strategy.sharding_configs = {
        "segment_anchors": None,
        "sharding_segment_strategy": "segment_broadcast_MB",
        "segment_broadcast_MB": 32,
        "sharding_degree": int(paddle.distributed.get_world_size()),
    }
    optimizer = fleet.distributed_optimizer(optimizer, dist_strategy)
    optimizer.minimize(loss)

    place = paddle.CUDAPlace(paddle.distributed.ParallelEnv().dev_id)
    exe = paddle.static.Executor(place)
    exe.run(paddle.static.default_startup_program())

    # bind gen
    train_ds = ShardedDataset(graph.nodes)
    collate_fn = BatchRandWalk(graph, args.walk_len, args.win_size,
                               args.neg_num, args.neg_sample_type)
    data_loader = Dataloader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.sample_workers,
        collate_fn=collate_fn)

    for epoch in range(args.epoch):
        train_loss = train(exe,
                           paddle.static.default_main_program(), data_loader,
                           loss)
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
    args = parser.parse_args()

    # merge user args and config file 
    config = edict(yaml.load(open(args.conf), Loader=yaml.FullLoader))
    config.update(vars(args))
    main(config)
