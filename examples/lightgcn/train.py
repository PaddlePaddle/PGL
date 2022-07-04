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

import pgl
import paddle
import paddle.nn as nn
from dataloader import Loader
from model import *
from utils import *
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import time
import multiprocessing
from tqdm import tqdm
from time import time


def train(dataset, model, epoch, optim, args, neg_k=1, w=None):
    model.train()
    with timer(name="Sample"):
        S = UniformSample_original_python(dataset)
    users, posItems, negItems = shuffle(S[:, 0], S[:, 1], S[:, 2])
    users = paddle.to_tensor(users, dtype='int64')
    posItems = paddle.to_tensor(posItems, dtype='int64')
    negItems = paddle.to_tensor(negItems, dtype='int64')
    batch_size = int(args.batch_size)
    total_batch = len(users) // batch_size + 1
    avg_loss = 0.
    pbar = tqdm(minibatch(users, posItems, negItems, batch_size=batch_size))
    for (batch_i, (batch_users, batch_pos, batch_neg)) in enumerate(pbar):
        loss, reg_loss = model.bpr_loss(batch_users, batch_pos, batch_neg)
        reg_loss = reg_loss * args.decay
        loss = loss + reg_loss
        loss.backward()
        optim.step()
        optim.clear_grad()
        avg_loss += loss.numpy()
        pbar.set_description(f'losses: {avg_loss[0]/(batch_i+1)}')

    avg_loss = avg_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{avg_loss[0]:.3f}-{time_info}"


def test_one_batch(X, topks):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in topks:
        ret = RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(NDCGatK_r(groundTrue, r, k))
    return {
        'recall': np.array(recall),
        'precision': np.array(pre),
        'ndcg': np.array(ndcg)
    }


@paddle.no_grad()
def test(dataset, model, epoch, args):
    u_batch_size = args.test_batch_size
    testDict = dataset.testDict
    model.eval()
    topks = eval(args.topks)
    max_K = max(topks)
    results = {
        'precision': np.zeros(len(topks)),
        'recall': np.zeros(len(topks)),
        'ndcg': np.zeros(len(topks))
    }
    users = list(testDict.keys())
    users_list = []
    rating_list = []
    groundTrue_list = []
    total_batch = len(users) // u_batch_size + 1
    for batch_users in tqdm(minibatch(users, batch_size=u_batch_size)):
        allPos = dataset.getUserPosItems(batch_users)
        groundTrue = [testDict[u] for u in batch_users]
        batch_users_gpu = paddle.to_tensor(batch_users, dtype='int64')
        rating = model.getUsersRating(batch_users_gpu)
        exclude_index = []
        exclude_items = []

        for range_i, items in enumerate(allPos):
            exclude_index.extend([range_i] * len(items))
            items = [int(i) for i in items]
            exclude_items.extend(items)
        rating[exclude_index, exclude_items] = -(1 << 10)
        _, rating_K = paddle.topk(rating, k=max_K)
        rating = rating.numpy()
        del rating
        users_list.append(batch_users)
        rating_list.append(rating_K.cpu())
        groundTrue_list.append(groundTrue)

    assert total_batch == len(users_list)
    X = zip(rating_list, groundTrue_list)
    pre_results = []
    for x in X:
        pre_results.append(test_one_batch(x, topks))
    scale = float(u_batch_size / len(users))
    for result in pre_results:
        results['recall'] += result['recall']
        results['precision'] += result['precision']
        results['ndcg'] += result['ndcg']
    results['recall'] /= float(len(users))
    results['precision'] /= float(len(users))
    results['ndcg'] /= float(len(users))
    # results['auc'] = np.mean(auc_record)
    return results


import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1024,
        help="the batch size for bpr loss training procedure")
    parser.add_argument(
        '--recdim',
        type=int,
        default=64,
        help="the embedding size of lightGCN")
    parser.add_argument(
        '--n_layers', type=int, default=3, help="the layer num of lightGCN")
    parser.add_argument(
        '--lr', type=float, default=0.0001, help="the learning rate")
    parser.add_argument(
        '--decay',
        type=float,
        default=1e-5,
        help="the weight decay for l2 normalizaton")
    parser.add_argument(
        '--test_batch_size',
        type=int,
        default=1024,
        help="the batch size of users for testing")
    parser.add_argument(
        '--dataset',
        type=str,
        default='gowalla',
        help="available datasets: [lastfm, gowalla, yelp2018, amazon-book]")
    parser.add_argument(
        '--path',
        type=str,
        default="./checkpoints",
        help="path to save weights")
    parser.add_argument(
        '--topks', nargs='?', default="[20]", help="@k test list")
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    dataset = Loader(args, path=args.dataset)
    model = LightGCN(args, dataset)
    optim = paddle.optimizer.Adam(
        parameters=model.parameters(), learning_rate=args.lr)

    for epoch in range(args.epochs):
        if epoch % 10 == 0:
            results = test(dataset, model, epoch, args)
            print(results)

        log = train(dataset, model, epoch, optim, args, neg_k=1, w=None)
        print(f'EPOCH[{epoch+1}/{args.epochs}] {log}')
