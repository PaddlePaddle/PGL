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
import os
import sys
import time
import argparse
import numpy as np

import paddle
import paddle.nn as nn
from paddle.optimizer import Adam
import pgl
from pgl.utils.logger import log
from pgl.utils.data import Dataloader
from dataset import GINDataset, fold10_split, random_split, collate_fn
from model import GINModel


def main(args):
    """
    train entry
    """
    ds = GINDataset(
        args.data_path,
        args.dataset_name,
        self_loop=not args.train_eps,
        degree_as_nlabel=True)
    args.num_nodes = int(ds.n / ds.num_graph)
    args.feat_size = ds.dim_nfeats

    res = []
    for fold in range(10):
        train_ds, test_ds = fold10_split(ds, fold_idx=fold, seed=args.seed)
        train_loader = Dataloader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=1,
            collate_fn=collate_fn)
        test_loader = Dataloader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=1,
            collate_fn=collate_fn)
        model = GINModel(args, ds.gclasses)
        epoch_step = len(train_loader)
        boundaries = [
            i
            for i in range(50 * epoch_step, args.epochs * epoch_step,
                           epoch_step * 50)
        ]
        values = [args.lr * 0.5**i for i in range(0, len(boundaries) + 1)]
        scheduler = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=boundaries, values=values, verbose=False)
        optim = Adam(
            learning_rate=scheduler,
            parameters=model.parameters(),
            weight_decay=args.weight_decay)
        criterion = nn.loss.CrossEntropyLoss()
        global_step = 0
        best_acc = 0.0

        for epoch in range(1, args.epochs + 1):
            model.train()
            for idx, batch_data in enumerate(train_loader):
                graphs, labels = batch_data
                g = pgl.Graph.batch(graphs).tensor()
                labels = paddle.to_tensor(labels)

                pred = model(g)
                train_loss = criterion(pred, labels)
                train_loss.backward()
                train_acc = paddle.metric.accuracy(
                    input=pred, label=labels, k=1)
                optim.step()
                optim.clear_grad()
                scheduler.step()

                global_step += 1
                if global_step % 10 == 0:
                    message = "train: epoch %d | step %d | " % (epoch,
                                                                global_step)
                    message += "loss %.6f | acc %.4f" % (train_loss, train_acc)
                    log.info(message)

            result = evaluate(model, test_loader, criterion)
            message = "eval: epoch %d | step %d | " % (epoch, global_step)
            for key, value in result.items():
                message += " | %s %.6f" % (key, value)
            log.info(message)

            if best_acc < result['acc']:
                best_acc = result['acc']
        res.append(best_acc)

    with open("outputs/" + args.dataset_name + "-" + ".log", "a") as f:
        f.write(str(args) + "\n")
        f.write(str(res) + "\n")
        f.write(str(sum(res) / 10) + '\n')

    print(res)
    log.info("best evaluating accuracy: " + str(sum(res) / 10))


def evaluate(model, loader, criterion):
    """
    Evaluate entry
    """
    model.eval()
    total_loss = []
    total_acc = []

    for idx, batch_data in enumerate(loader):
        graphs, labels = batch_data
        g = pgl.Graph.batch(graphs).tensor()
        labels = paddle.to_tensor(labels)

        pred = model(g)
        eval_loss = criterion(pred, labels)
        eval_acc = paddle.metric.accuracy(input=pred, label=labels, k=1)
        total_loss.append(eval_loss.numpy())
        total_acc.append(eval_acc.numpy())

    total_loss = np.mean(total_loss)
    total_acc = np.mean(total_acc)
    model.train()

    return {"loss": total_loss, "acc": total_acc}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./gin_data')
    parser.add_argument('--dataset_name', type=str, default='MUTAG')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--fold_idx', type=int, default=0)
    parser.add_argument('--output_path', type=str, default='./outputs/')
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--num_mlp_layers', type=int, default=2)
    parser.add_argument('--feat_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--train_eps', action='store_true')
    parser.add_argument('--init_eps', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--min_score', type=float)
    parser.add_argument('--pool_ratio', type=float, default=0.15)
    args = parser.parse_args()

    log.info(args)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    main(args)
