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


def train(dataloader, model, feature, criterion, optim, log_per_step=100):
    model.train()

    batch = 0
    total_loss = 0.
    total_acc = 0.
    total_sample = 0

    for g, sample_index, index, label in dataloader:
        batch += 1
        num_samples = len(index)

        g.tensor()
        sample_index = paddle.to_tensor(sample_index)
        index = paddle.to_tensor(index)
        label = paddle.to_tensor(label)

        feat = paddle.gather(feature, sample_index)
        pred = model(g, feat)
        pred = paddle.gather(pred, index)
        loss = criterion(pred, label)
        loss.backward()
        acc = paddle.metric.accuracy(input=pred, label=label, k=1)
        optim.step()
        optim.clear_grad()

        total_loss += loss.numpy() * num_samples
        total_acc += acc.numpy() * num_samples
        total_sample += num_samples

        if batch % log_per_step == 0:
            log.info("Batch %s %s-Loss %s %s-Acc %s" %
                     (batch, "train", loss.numpy(), "train", acc.numpy()))

    return total_loss / total_sample, total_acc / total_sample


@paddle.no_grad()
def eval(dataloader, model, feature, criterion):
    model.eval()
    loss_all, acc_all = [], []
    for g, sample_index, index, label in dataloader:
        g.tensor()
        sample_index = paddle.to_tensor(sample_index)
        index = paddle.to_tensor(index)
        label = paddle.to_tensor(label)

        feat = paddle.gather(feature, sample_index)
        pred = model(g, feat)
        pred = paddle.gather(pred, index)
        loss = criterion(pred, label)
        acc = paddle.metric.accuracy(input=pred, label=label, k=1)
        loss_all.append(loss.numpy())
        acc_all.append(acc.numpy())

    return np.mean(loss_all), np.mean(acc_all)


def main(args):
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

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

    model = GraphSage(
        input_size=data.feature.shape[-1],
        num_class=data.num_classes,
        hidden_size=args.hidden_size,
        num_layers=len(args.samples))

    model = paddle.DataParallel(model)

    criterion = paddle.nn.loss.CrossEntropyLoss()

    optim = paddle.optimizer.Adam(
        learning_rate=args.lr,
        parameters=model.parameters(),
        weight_decay=0.001)

    feature = paddle.to_tensor(data.feature)

    train_ds = ShardedDataset(train_index, train_label)
    val_ds = ShardedDataset(val_index, val_label)
    test_ds = ShardedDataset(test_index, test_label)

    collate_fn = partial(batch_fn, graph=graph, samples=args.samples)

    train_loader = Dataloader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.sample_workers,
        collate_fn=collate_fn)
    val_loader = Dataloader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.sample_workers,
        collate_fn=collate_fn)
    test_loader = Dataloader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.sample_workers,
        collate_fn=collate_fn)

    cal_val_acc = []
    cal_test_acc = []
    cal_val_loss = []
    for epoch in tqdm.tqdm(range(args.epoch)):
        train_loss, train_acc = train(train_loader, model, feature, criterion,
                                      optim)
        log.info("Runing epoch:%s\t train_loss:%s\t train_acc:%s", epoch,
                 train_loss, train_acc)
        val_loss, val_acc = eval(val_loader, model, feature, criterion)
        cal_val_acc.append(val_acc)
        cal_val_loss.append(val_loss)
        log.info("Runing epoch:%s\t val_loss:%s\t val_acc:%s", epoch, val_loss,
                 val_acc)
        test_loss, test_acc = eval(test_loader, model, feature, criterion)
        cal_test_acc.append(test_acc)
        log.info("Runing epoch:%s\t test_loss:%s\t test_acc:%s", epoch,
                 test_loss, test_acc)

    log.info("Runs %s: Model: %s Best Test Accuracy: %f" %
             (0, "graphsage", cal_test_acc[np.argmax(cal_val_acc)]))


if __name__ == "__main__":
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
    log.info(args)
    main(args)
