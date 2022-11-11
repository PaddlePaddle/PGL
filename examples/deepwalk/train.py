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
from paddle.io import get_worker_info

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


def load_from_file(path):
    edges = []
    with open(path) as inf:
        for line in inf:
            u, t = line.strip("\n").split("\t")
            u, t = int(u), int(t)
            edges.append((u, t))
    edges = np.array(edges)
    graph = pgl.Graph(edges)
    return graph


def train(model, data_loader, optim, log_per_step=10):
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

        total_loss += float(loss) * num_samples
        total_sample += num_samples

        if batch % log_per_step == 0:
            log.info("Batch %s %s-Loss %.6f" % (batch, "train", float(loss)))

    return total_loss / total_sample


def main(args):
    if not args.use_cuda:
        paddle.set_device("cpu")
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    if args.edge_file:
        graph = load_from_file(args.edge_file)
    else:
        graph = load(args.dataset)

    model = SkipGramModel(
        graph.num_nodes,
        args.embed_size,
        args.neg_num,
        sparse=not args.use_cuda,
        shared_embedding=args.shared_embedding)
    model = paddle.DataParallel(model)

    train_ds = ShardedDataset(graph.nodes, repeat=args.epoch)

    train_steps = int(len(train_ds) // args.batch_size)
    log.info("train_steps: %s" % train_steps)
    scheduler = paddle.optimizer.lr.PolynomialDecay(
        learning_rate=args.learning_rate,
        decay_steps=train_steps,
        end_lr=0.0001)

    optim = Adam(learning_rate=scheduler, parameters=model.parameters())

    collate_fn = BatchRandWalk(graph, args.walk_len, args.win_size,
                               args.neg_num, args.neg_sample_type)
    data_loader = Dataloader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.sample_workers,
        collate_fn=collate_fn)

    train_loss = train(model, data_loader, optim)
    paddle.save(model.state_dict(), "model.pdparams")


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
    parser.add_argument("--epoch", type=int, default=400, help="Epoch")
    parser.add_argument("--edge_file", type=str, default=None)
    args = parser.parse_args()

    # merge user args and config file 
    config = edict(yaml.load(open(args.conf), Loader=yaml.FullLoader))
    config.update(vars(args))
    main(config)
