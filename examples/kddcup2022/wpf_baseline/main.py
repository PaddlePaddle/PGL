# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import tqdm
import yaml
import numpy as np
from easydict import EasyDict as edict

import pgl
from pgl.utils.logger import log
from paddle.io import DataLoader
import random

import models
import loss as loss_factory
from wpf_dataset import PGL4WPFDataset
import optimization as optim
from metrics import regressor_metrics
from utils import save_model, _create_if_not_exist, load_model
import matplotlib.pyplot as plt


def data_augment(X, y, p=0.8, alpha=0.5, beta=0.5):
    """Regression SMOTE
    """
    fix_X, X = X[:, :, :, :2], X[:, :, :, 2:]
    fix_y, y = y[:, :, :, :2], y[:, :, :, 2:]
    batch_size = X.shape[0]
    random_values = paddle.rand([batch_size])
    idx_to_change = random_values < p

    # ensure that first element to switch has probability > 0.5
    np_betas = np.random.beta(alpha, beta, batch_size) / 2 + 0.5
    random_betas = paddle.to_tensor(
        np_betas, dtype="float32").reshape([-1, 1, 1, 1])
    index_permute = paddle.randperm(batch_size)

    X[idx_to_change] = random_betas[idx_to_change] * X[idx_to_change]
    X[idx_to_change] += (
        1 - random_betas[idx_to_change]) * X[index_permute][idx_to_change]

    y[idx_to_change] = random_betas[idx_to_change] * y[idx_to_change]
    y[idx_to_change] += (
        1 - random_betas[idx_to_change]) * y[index_permute][idx_to_change]
    return paddle.concat([fix_X, X], -1), paddle.concat([fix_y, y], -1)


def train_and_evaluate(config, train_data, valid_data, test_data):
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    data_mean = paddle.to_tensor([train_data.data_mean], dtype="float32")
    data_scale = paddle.to_tensor([train_data.data_scale], dtype="float32")

    graph = train_data.graph
    graph = graph.tensor()

    train_data_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers)

    valid_data_loader = DataLoader(
        valid_data,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers)

    test_data_loader = DataLoader(
        test_data,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers)

    model = getattr(models, config.model.name).WPFModel(config=config)

    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    loss_fn = getattr(loss_factory, config.loss.name)(
        **dict(config.loss.items()))

    opt = optim.get_optimizer(model=model, learning_rate=config.lr)

    _create_if_not_exist(config.output_path)
    global_step = 0

    best_score = np.inf
    patient = 0

    valid_records = []
    test_records = []

    for epoch in range(config.epoch):
        for batch_x, batch_y in train_data_loader:
            batch_x = batch_x.astype('float32')
            batch_y = batch_y.astype('float32')
            batch_x, batch_y = data_augment(batch_x, batch_y)
            input_len = batch_x.shape[1]

            input_y = batch_y
            batch_y = batch_y[:, :, :, -1]
            batch_y = (batch_y - data_mean) / data_scale
            pred_y = model(batch_x, input_y, data_mean, data_scale, graph)
            loss = loss_fn(pred_y, batch_y)
            loss.backward()

            opt.step()
            opt.clear_gradients()
            global_step += 1
            if paddle.distributed.get_rank(
            ) == 0 and global_step % config.log_per_steps == 0:
                log.info("Step %s Train MSE-Loss: %s RMSE-Loss: %s" %
                         (global_step, loss.numpy()[0],
                          (paddle.sqrt(loss)).numpy()[0]))

        if paddle.distributed.get_rank() == 0:

            valid_r, valid_pred, valid_label = evaluate(
                valid_data_loader,
                model,
                loss_fn,
                config,
                data_mean,
                data_scale,
                tag="val",
                graph=graph)
            valid_records.append(valid_r)

            log.info("Valid " + str(dict(valid_r)))

            best_score = min(valid_r['score'], best_score)

            test_r, test_pred, test_label = evaluate(
                test_data_loader,
                model,
                loss_fn,
                config,
                data_mean,
                data_scale,
                tag="test",
                graph=graph)
            log.info("Test " + str(dict(test_r)))
            test_records.append(test_r)

            if best_score == valid_r['score']:
                patient = 0
                save_model(config.output_path, model, steps=epoch)
            else:
                patient += 1
                if patient > config.patient:
                    break

    best_epochs = max(enumerate(valid_records), key=lambda x: x[1]["score"])[0]
    log.info("Best valid Epoch %s" % best_epochs)
    log.info("Best valid score %s" % valid_records[best_epochs])
    log.info("Best valid test-score %s" % test_records[best_epochs])


def visualize_prediction(input_batch, pred_batch, gold_batch, tag):
    plt.figure()
    for i in range(1, 5):
        ax = plt.subplot(2, 2, i)
        ax.plot(
            np.concatenate(
                [input_batch[288 * (i - 1)], gold_batch[288 * (i - 1)]]),
            label="gold")
        ax.plot(
            np.concatenate(
                [input_batch[288 * (i - 1)], pred_batch[288 * (i - 1)]]),
            label="pred")
        ax.legend()
    plt.savefig(tag + "_vis.png")


@paddle.no_grad()
def evaluate(valid_data_loader,
             model,
             loss_fn,
             config,
             data_mean,
             data_scale,
             tag="train",
             graph=None):
    model.eval()
    step = 0
    pred_batch = []
    gold_batch = []
    input_batch = []
    losses = []
    for batch_x, batch_y in valid_data_loader:
        batch_x = batch_x.astype('float32')
        batch_y = batch_y.astype('float32')
        input_len = batch_x.shape[1]

        pred_y = model(batch_x, batch_y, data_mean, data_scale, graph)

        scaled_batch_y = batch_y[:, :, :, -1]
        scaled_batch_y = (scaled_batch_y - data_mean) / data_scale
        loss = loss_fn(pred_y, scaled_batch_y)
        losses.append(loss.numpy()[0])

        pred_y = pred_y * data_scale + data_mean
        pred_y = paddle.sum(pred_y, 1).numpy()

        batch_y = paddle.sum(batch_y[:, :, :, -1], axis=1).numpy()

        input_batch.append(paddle.sum(batch_x[:, :, :, -1], axis=1).numpy())
        pred_batch.append(pred_y)
        gold_batch.append(batch_y)

        step += 1
    model.train()

    pred_batch = np.concatenate(pred_batch, axis=0) / 1000
    gold_batch = np.concatenate(gold_batch, axis=0) / 1000
    input_batch = np.concatenate(input_batch, axis=0) / 1000

    visualize_prediction(input_batch, pred_batch, gold_batch, tag)

    _mae, _mse, _rmse = regressor_metrics(pred_batch, gold_batch)
    output_metric = {
        'mae': _mae,
        'mse': _mse,
        'score': (_mae + _rmse) / 2,
        'rmse': _rmse,
        'loss': np.mean(losses),
    }

    return output_metric, pred_batch, gold_batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--conf", type=str, default="./config.yaml")
    args = parser.parse_args()
    config = edict(yaml.load(open(args.conf), Loader=yaml.FullLoader))

    print(config)
    size = [config.input_len, config.output_len]
    train_data = PGL4WPFDataset(
        config.data_path, filename="wtb5_10.csv", size=size, flag='train')
    valid_data = PGL4WPFDataset(
        config.data_path, filename="wtb5_10.csv", size=size, flag='val')
    test_data = PGL4WPFDataset(
        config.data_path, filename="wtb5_10.csv", size=size, flag='test')

    train_and_evaluate(config, train_data, valid_data, test_data)
