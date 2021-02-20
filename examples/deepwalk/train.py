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
import time
import argparse

import pgl
import paddle
import paddle.nn as nn
from pgl.utils.logger import log
import numpy as np
import yaml
from easydict import EasyDict as edict
import tqdm
from paddle.optimizer import Adam
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
    return dataset.graph.to_mmap()


def train(model, data_loader, optim, log_per_step=1):
    model.train()
    total_loss = 0.
    total_sample = 0

    for batch, (src, dsts) in enumerate(data_loader):
        num_samples = len(src)
        src = paddle.to_tensor(src)
        dsts = paddle.to_tensor(dsts)
        loss = model(src, dsts)
        loss.backward()
        optim.step()
        optim.clear_grad()

        total_loss += loss.numpy()[0] * num_samples
        total_sample += num_samples

        if batch % log_per_step == 0:
            log.info("Batch %s %s-Loss %.6f" %
                     (batch, "train", loss.numpy()[0]))

    return total_loss / total_sample


def main(args):
    if not args.use_cuda:
        paddle.set_device("cpu")
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    graph = load(args.dataset)

    model = SkipGramModel(
        graph.num_nodes,
        args.embed_size,
        args.neg_num,
        sparse=not args.use_cuda)
    model = paddle.DataParallel(model)

    optim = Adam(
        learning_rate=args.learning_rate,
        parameters=model.parameters())

    train_ds = ShardedDataset(graph.nodes)
    collate_fn = BatchRandWalk(graph, args.walk_len, args.win_size,
                               args.neg_num, args.neg_sample_type)
    data_loader = Dataloader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.sample_workers,
        collate_fn=collate_fn)

    for epoch in tqdm.tqdm(range(args.epoch)):
        train_loss = train(model, data_loader, optim)
        log.info("Runing epoch:%s\t train_loss:%.6f", epoch, train_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deepwalk')
    parser.add_argument(
        "--dataset",
        type=str,
        default="BlogCatalog",
        help="dataset (cora, pubmed, BlogCatalog)")
    parser.add_argument("--use_cuda", action='store_true', help="use_cuda")
    parser.add_argument(
        "--conf",
        type=str,
        default="./config.yaml",
        help="config file for models")
    parser.add_argument("--epoch", type=int, default=200, help="Epoch")
    args = parser.parse_args()

    # merge user args and config file 
    config = edict(yaml.load(open(args.conf), Loader=yaml.FullLoader))
    config.update(vars(args))
    main(config)
