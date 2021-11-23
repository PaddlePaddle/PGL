# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import argparse
from functools import partial

import yaml
from tqdm import tqdm
from easydict import EasyDict as edict
import numpy as np
import paddle
import pgl
from pgl.utils.logger import log

sys.path.insert(0, os.path.abspath(".."))
import gnn_models
from dataset import load_dataset, create_dataloaders
from graph_partition import random_graph_partition
from graph_partition import metis_graph_partition
from utils import check_device, process_batch_data, compute_buffer_size
from utils import generate_mask, permute, compute_gcn_norm, compute_acc


def train_eval(model, feature, gcn_norm, criterion, optim, dataset, data_mode,
               eval_graph, best_val_acc, final_test_acc, batch_nums, log_epoch,
               train_loader, eval_loader):
    model.train()

    batch_id = 0
    total_loss = total_examples = 0

    if isinstance(train_loader, list):
        # If we get a list-type train_loader, that means we generate train data in advance.
        # Then we need to shuffle this list before each epoch starts.
        np.random.shuffle(train_loader)

    for batch_data in train_loader:
        batch_id += 1
        g, batch_size, n_id, offset, count, feat, sub_norm = \
            process_batch_data(batch_data, feature, gcn_norm)
        pred = model(g, feat, sub_norm, batch_size, n_id, offset, count)
        pred = pred[:batch_size]

        sub_train_mask = paddle.gather(
            paddle.to_tensor(dataset.train_mask), n_id[:batch_size])
        y = paddle.gather(paddle.to_tensor(dataset.label), n_id[:batch_size])
        true_index = paddle.nonzero(sub_train_mask)
        if true_index.shape[0] == 0:
            continue
        pred = paddle.gather(pred, true_index)
        y = paddle.gather(y, true_index)

        loss = criterion(pred, y)
        loss.backward()
        optim.step()
        optim.clear_grad()

        total_loss += loss.numpy() * true_index.shape[0]
        total_examples += true_index.shape[0]

        if batch_id % batch_nums == 0:
            # Finish one training epoch, print evaluation results.
            log_epoch += 1
            train_acc, val_acc, test_acc, best_val_acc, final_test_acc = run_eval(
                data_mode, dataset, eval_graph, model, feature, gcn_norm,
                best_val_acc, final_test_acc, eval_loader)
            log.info(f"Epoch: %d, Train Loss: %.4f, Train Acc: %.4f, "
                     f"Valid Acc: %.4f, Test Acc: %.4f, Final Acc: %.4f" %
                     (log_epoch, loss, train_acc, val_acc, test_acc,
                      final_test_acc))
            model.train()
            batch_id = 0

    return best_val_acc, final_test_acc, log_epoch


@paddle.no_grad()
def full_eval(graph, model, feature, norm):
    # For small graph, we can get full-batch inference result.
    model.eval()
    feature = paddle.to_tensor(feature)
    if norm is not None:
        norm = paddle.to_tensor(norm)
    out = model(graph, feature, norm)
    return out


@paddle.no_grad()
def mini_eval(graph, model, feature, norm, eval_loader):
    model.eval()
    out = model(graph, feature, norm, loader=eval_loader)
    return out


@paddle.no_grad()
def run_eval(mode,
             dataset,
             graph,
             model,
             feature,
             gcn_norm,
             best_val_acc,
             final_test_acc,
             eval_loader=None):
    if mode == 's':
        out = full_eval(graph, model, feature, gcn_norm)
    elif mode == 'm':
        out = mini_eval(graph, model, feature, gcn_norm, eval_loader)
    else:
        raise ValueError("Mode %s not supported." % mode)
    train_acc = compute_acc(out, dataset.label, dataset.train_mask)
    val_acc = compute_acc(out, dataset.label, dataset.val_mask)
    test_acc = compute_acc(out, dataset.label, dataset.test_mask)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        final_test_acc = test_acc
    return train_acc, val_acc, test_acc, best_val_acc, final_test_acc


def main(args, config):
    log.info("Loading %s dataset." % config.data_name)
    dataset, data_mode = load_dataset(config.data_name)

    log.info("Running into %d %s graph partitions." %
             (config.num_parts, args.partition))
    if args.partition == 'metis':
        permutation, part = metis_graph_partition(
            dataset.graph, npart=config.num_parts)
    elif args.partition == 'random':
        permutation, part = random_graph_partition(
            dataset.graph, npart=config.num_parts)
    else:
        raise ValueError("%s graph partition methods not supported" %
                         args.partition)

    log.info("Permuting dataset and feature.")
    dataset, feature = permute(dataset, dataset.feature, permutation,
                               config.feat_gpu)
    graph = dataset.graph

    log.info("Building data loader for training and inference.")
    train_loader, eval_loader, final_epochs = \
        create_dataloaders(graph, data_mode, part, args.num_workers,
                           config, args.load_epoch)
    gcn_norm = compute_gcn_norm(graph, config.gcn_norm)

    log.info("Calculating buffer size for GNNAutoScale.")
    buffer_size = compute_buffer_size(eval_loader)
    log.info("Buffer size: %d" % buffer_size)

    GNNModel = getattr(gnn_models, config.model_name)
    model = GNNModel(
        num_nodes=graph.num_nodes,
        input_size=feature.shape[1],
        output_size=dataset.num_classes,
        buffer_size=buffer_size,
        **config)

    if config.grad_norm:
        clip = paddle.nn.ClipGradByNorm(clip_norm=1.0)
    else:
        clip = None
    optim = paddle.optimizer.Adam(
        learning_rate=config.lr,
        weight_decay=config.weight_decay,
        parameters=model.parameters(),
        grad_clip=clip)

    criterion = paddle.nn.loss.CrossEntropyLoss()

    if data_mode == 's':
        eval_graph = pgl.Graph(edges=graph.edges, num_nodes=graph.num_nodes)
        eval_graph.tensor()
    else:
        eval_graph = graph
    best_val_acc = final_test_acc = log_epoch = 0
    num_batches = config.num_parts / config.batch_size
    for epoch in range(final_epochs):
        best_val_acc, final_test_acc, log_epoch = train_eval(
            model, feature, gcn_norm, criterion, optim, dataset, data_mode,
            eval_graph, best_val_acc, final_test_acc, num_batches, log_epoch,
            train_loader, eval_loader)

    log.info("Final Valid Acc: %.4f, Final Test Acc: %.4f" %
             (best_val_acc, final_test_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Main program for running GNNAutoScale.')
    parser.add_argument(
        "--conf", type=str, help="Config file for running gnn models.")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="Number of workers for train dataloader.")
    parser.add_argument(
        "--partition",
        type=str,
        default='metis',
        help=f"Set graph partition methods, including `metis` and `random`. "
        f"Note that we do not support metis partition on Windows system currently."
    )
    parser.add_argument(
        "--load_epoch",
        type=int,
        default=100,
        help=f"Mainly used in creating train dataset, in order to speed up the "
        f"data loading process and reduce process switching. It is useful "
        f"for dataset with small number of batches.")
    args = parser.parse_args()
    config = edict(yaml.load(open(args.conf), Loader=yaml.FullLoader))

    if not check_device():
        exit()

    log.info(args)
    log.info(config)
    main(args, config)
