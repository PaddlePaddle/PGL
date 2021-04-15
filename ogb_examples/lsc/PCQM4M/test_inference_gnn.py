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
    test_ds = Subset(ds, split_idx['test'])

    print("Test exapmles: ", len(test_ds))

    ### automatic evaluator. takes dataset name as input
    evaluator = PCQM4MEvaluator()

    test_loader = Dataloader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=CollateFn())

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

    checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint.pdparams')
    if not os.path.exists(checkpoint_path):
        raise RuntimeError(f'Checkpoint file not found at {checkpoint_path}')

    model.set_state_dict(paddle.load(checkpoint_path))

    print('Predicting on test data...')
    y_pred = test(model, test_loader)
    print('Saving test submission file...')
    evaluator.save_test_submission({'y_pred': y_pred}, args.save_test_dir)


if __name__ == "__main__":
    main()
