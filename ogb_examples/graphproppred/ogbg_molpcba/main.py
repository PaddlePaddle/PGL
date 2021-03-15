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

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
import sys
import time
import argparse
import numpy as np
from datetime import datetime

from ogb.graphproppred import GraphPropPredDataset, Evaluator
from tensorboardX import SummaryWriter

import paddle
import paddle.nn as nn
from paddle.optimizer import Adam
import paddle.distributed as dist

import pgl
from pgl.utils.logger import log
from pgl.utils.data import Dataloader

from model import ClassifierNetwork
from dataset import Subset, MolDataset, make_multihop_edges, CollateFn
from utils.config import prepare_config, make_dir
from utils.logger import prepare_logger, log_to_file


def main(config):
    if dist.get_world_size() > 1:
        dist.init_parallel_env()

    if dist.get_rank() == 0:
        timestamp = datetime.now().strftime("%Hh%Mm%Ss")
        log_path = os.path.join(config.log_dir,
                                "tensorboard_log_%s" % timestamp)
        writer = SummaryWriter(log_path)

    log.info("loading data")
    raw_dataset = GraphPropPredDataset(name=config.dataset_name)
    config.num_class = raw_dataset.num_tasks
    config.eval_metric = raw_dataset.eval_metric
    config.task_type = raw_dataset.task_type

    mol_dataset = MolDataset(
        config, raw_dataset, transform=make_multihop_edges)
    splitted_index = raw_dataset.get_idx_split()
    train_ds = Subset(mol_dataset, splitted_index['train'], mode='train')
    valid_ds = Subset(mol_dataset, splitted_index['valid'], mode="valid")
    test_ds = Subset(mol_dataset, splitted_index['test'], mode="test")

    log.info("Train Examples: %s" % len(train_ds))
    log.info("Val Examples: %s" % len(valid_ds))
    log.info("Test Examples: %s" % len(test_ds))

    fn = CollateFn(config)

    train_loader = Dataloader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=fn)

    valid_loader = Dataloader(
        valid_ds,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=fn)

    test_loader = Dataloader(
        test_ds,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=fn)

    model = ClassifierNetwork(config.hidden_size, config.out_dim,
                              config.num_layers, config.dropout_prob,
                              config.virt_node, config.K, config.conv_type,
                              config.appnp_hop, config.alpha)
    model = paddle.DataParallel(model)

    optim = Adam(learning_rate=config.lr, parameters=model.parameters())
    criterion = nn.loss.BCEWithLogitsLoss()

    evaluator = Evaluator(config.dataset_name)

    best_valid = 0

    global_step = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        for idx, batch_data in enumerate(train_loader):
            g, mh_graphs, labels, unmask = batch_data
            g = g.tensor()
            multihop_graphs = []
            for item in mh_graphs:
                multihop_graphs.append(item.tensor())
            g.multi_hop_graphs = multihop_graphs
            labels = paddle.to_tensor(labels)
            unmask = paddle.to_tensor(unmask)

            pred = model(g)
            pred = paddle.masked_select(pred, unmask)
            labels = paddle.masked_select(labels, unmask)
            train_loss = criterion(pred, labels)
            train_loss.backward()
            optim.step()
            optim.clear_grad()

            if global_step % 80 == 0:
                message = "train: epoch %d | step %d | " % (epoch, global_step)
                message += "loss %.6f" % (train_loss.numpy())
                log.info(message)
                if dist.get_rank() == 0:
                    writer.add_scalar("loss", train_loss.numpy(), global_step)
            global_step += 1

        valid_result = evaluate(model, valid_loader, criterion, evaluator)
        message = "valid: epoch %d | step %d | " % (epoch, global_step)
        for key, value in valid_result.items():
            message += " | %s %.6f" % (key, value)
            if dist.get_rank() == 0:
                writer.add_scalar("valid_%s" % key, value, global_step)
        log.info(message)

        test_result = evaluate(model, test_loader, criterion, evaluator)
        message = "test: epoch %d | step %d | " % (epoch, global_step)
        for key, value in test_result.items():
            message += " | %s %.6f" % (key, value)
            if dist.get_rank() == 0:
                writer.add_scalar("test_%s" % key, value, global_step)
        log.info(message)

        if best_valid < valid_result[config.metrics]:
            best_valid = valid_result[config.metrics]
            best_valid_result = valid_result
            best_test_result = test_result

        message = "best result: epoch %d | " % (epoch)
        message += "valid %s: %.6f | " % (config.metrics,
                                          best_valid_result[config.metrics])
        message += "test %s: %.6f | " % (config.metrics,
                                         best_test_result[config.metrics])
        log.info(message)

    message = "final eval best result:%.6f" % best_valid_result[config.metrics]
    log.info(message)
    message = "final test best result:%.6f" % best_test_result[config.metrics]
    log.info(message)


@paddle.no_grad()
def evaluate(model, loader, criterion, evaluator):
    model.eval()
    total_loss = []
    y_true = []
    y_pred = []
    is_valid = []

    for idx, batch_data in enumerate(loader):
        g, mh_graphs, labels, unmask = batch_data
        g = g.tensor()
        multihop_graphs = []
        for item in mh_graphs:
            multihop_graphs.append(item.tensor())
        g.multi_hop_graphs = multihop_graphs
        labels = paddle.to_tensor(labels)
        unmask = paddle.to_tensor(unmask)

        pred = model(g)
        eval_loss = criterion(
            paddle.masked_select(pred, unmask),
            paddle.masked_select(labels, unmask))
        total_loss.append(eval_loss.numpy())

        y_pred.append(pred.numpy())
        y_true.append(labels.numpy())
        is_valid.append(unmask.numpy())

    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    is_valid = np.concatenate(is_valid)
    is_valid = is_valid.astype("bool")
    y_true = y_true.astype("float32")
    y_true[~is_valid] = np.nan
    input_dict = {'y_true': y_true, 'y_pred': y_pred}
    result = evaluator.eval(input_dict)

    total_loss = np.mean(total_loss)
    model.train()

    return {"loss": total_loss, config.metrics: result[config.metrics]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='gnn')
    parser.add_argument("--config", type=str, default="./config.yaml")
    parser.add_argument("--task_name", type=str, default="task_name")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--log_id", type=str, default=None)
    args = parser.parse_args()

    if dist.get_rank() == 0:
        config = prepare_config(args.config, isCreate=True, isSave=True)
        if args.log_id is not None:
            config.log_filename = "%s_%s" % (args.log_id, config.log_filename)
        log_to_file(log, config.log_dir, config.log_filename)
    else:
        config = prepare_config(args.config, isCreate=False, isSave=False)

    config.log_id = args.log_id
    main(config)
