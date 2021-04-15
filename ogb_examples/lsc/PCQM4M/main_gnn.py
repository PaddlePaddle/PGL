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
import time
import random
import argparse
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import pgl
from pgl.utils.data import Dataloader

### importing OGB-LSC
from ogb.lsc import PCQM4MDataset, PCQM4MEvaluator
from ogb.utils import smiles2graph

from dataset import MolDataset, Subset, CollateFn
from gnn import GNN

reg_criterion = paddle.nn.loss.L1Loss()


def train(model, loader, optimizer):
    model.train()
    loss_accum = 0

    for step, (g, labels) in enumerate(tqdm(loader, desc="Iteration")):
        g = g.tensor()
        labels = paddle.to_tensor(labels)

        pred = paddle.reshape(model(g), shape=[-1, ])
        loss = reg_criterion(pred, labels)
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        loss_accum += loss.numpy()

    return loss_accum / (step + 1)


@paddle.no_grad()
def eval(model, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, (g, labels) in enumerate(tqdm(loader, desc="Iteration")):
        g = g.tensor()
        #  labels = paddle.to_tensor(labels)

        pred = model(g)

        y_true.append(labels.reshape(-1, 1))
        y_pred.append(pred.numpy().reshape(-1, 1))

    y_true = np.concatenate(y_true).reshape(-1, )
    y_pred = np.concatenate(y_pred).reshape(-1, )

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)["mae"]


@paddle.no_grad()
def test(model, loader):
    model.eval()
    y_pred = []

    for step, (g, labels) in enumerate(tqdm(loader, desc="Iteration")):
        g = g.tensor()
        #  labels = paddle.to_tensor(labels)

        pred = model(g)

        y_pred.append(pred.numpy().reshape(-1, 1))

    y_pred = np.concatenate(y_pred).reshape(-1, )

    return y_pred


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='GNN baselines on pcqm4m with PGL')
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        help='which gpu to use if any (default: 0)')
    parser.add_argument(
        '--gnn',
        type=str,
        default='gin-virtual',
        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)'
    )
    parser.add_argument(
        '--graph_pooling',
        type=str,
        default='sum',
        help='graph pooling strategy mean or sum (default: sum)')
    parser.add_argument(
        '--drop_ratio',
        type=float,
        default=0,
        help='dropout ratio (default: 0)')
    parser.add_argument(
        '--num_layers',
        type=int,
        default=5,
        help='number of GNN message passing layers (default: 5)')
    parser.add_argument(
        '--emb_dim',
        type=int,
        default=600,
        help='dimensionality of hidden units in GNNs (default: 600)')
    parser.add_argument('--train_subset', action='store_true')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='input batch size for training (default: 256)')
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='number of epochs to train (default: 100)')
    parser.add_argument(
        '--num_workers',
        type=int,
        default=1,
        help='number of workers (default: 1)')
    parser.add_argument(
        '--log_dir', type=str, default="", help='tensorboard log directory')
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='',
        help='directory to save checkpoint')
    parser.add_argument(
        '--save_test_dir',
        type=str,
        default='',
        help='directory to save test submission file')
    args = parser.parse_args()

    print(args)

    random.seed(42)
    np.random.seed(42)
    paddle.seed(42)

    if not args.use_cuda:
        paddle.set_device("cpu")

    ### automatic dataloading and splitting
    class Config():
        def __init__(self):
            self.base_data_path = "./dataset"

    config = Config()
    ds = MolDataset(config)

    split_idx = ds.get_idx_split()
    train_ds = Subset(ds, split_idx['train'])
    valid_ds = Subset(ds, split_idx['valid'])
    test_ds = Subset(ds, split_idx['test'])

    print("Train exapmles: ", len(train_ds))
    print("Valid exapmles: ", len(valid_ds))
    print("Test exapmles: ", len(test_ds))

    ### automatic evaluator. takes dataset name as input
    evaluator = PCQM4MEvaluator()

    train_loader = Dataloader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=CollateFn())

    valid_loader = Dataloader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=CollateFn())

    if args.save_test_dir is not '':
        test_loader = Dataloader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=CollateFn())

    if args.checkpoint_dir is not '':
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    shared_params = {
        'num_layers': args.num_layers,
        'emb_dim': args.emb_dim,
        'drop_ratio': args.drop_ratio,
        'graph_pooling': args.graph_pooling
    }

    if args.gnn == 'gin':
        model = GNN(gnn_type='gin', virtual_node=False, **shared_params)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type='gin', virtual_node=True, **shared_params)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type='gcn', virtual_node=False, **shared_params)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type='gcn', virtual_node=True, **shared_params)
    else:
        raise ValueError('Invalid GNN type')

    num_params = sum(p.numel() for p in model.parameters())
    print(f'#Params: {num_params}')

    if args.log_dir is not '':
        writer = SummaryWriter(log_dir=args.log_dir)

    best_valid_mae = 1000

    scheduler = paddle.optimizer.lr.StepDecay(
        learning_rate=0.001, step_size=300, gamma=0.25)

    optimizer = paddle.optimizer.Adam(
        learning_rate=scheduler, parameters=model.parameters())

    msg = "ogbg_lsc_paddle_baseline\n"
    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train_mae = train(model, train_loader, optimizer)

        print('Evaluating...')
        valid_mae = eval(model, valid_loader, evaluator)

        print({'Train': train_mae, 'Validation': valid_mae})

        if args.log_dir is not '':
            writer.add_scalar('valid/mae', valid_mae, epoch)
            writer.add_scalar('train/mae', train_mae, epoch)

        if valid_mae < best_valid_mae:
            best_valid_mae = valid_mae
            if args.checkpoint_dir is not '':
                print('Saving checkpoint...')
                paddle.save(model.state_dict(),
                            os.path.join(args.checkpoint_dir,
                                         'checkpoint.pdparams'))

            if args.save_test_dir is not '':
                print('Predicting on test data...')
                y_pred = test(model, test_loader)
                print('Saving test submission file...')
                evaluator.save_test_submission({
                    'y_pred': y_pred
                }, args.save_test_dir)

        scheduler.step()

        print(f'Best validation MAE so far: {best_valid_mae}')

        try:
            msg +="Epoch: %d | Train: %.6f | Valid: %.6f | Best Valid: %.6f\n" \
                    % (epoch, train_mae, valid_mae, best_valid_mae)
            print(msg)
        except:
            continue

    if args.log_dir is not '':
        writer.close()


if __name__ == "__main__":
    main()
