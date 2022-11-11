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
import tqdm
import yaml
import argparse
import numpy as np

import pgl
from pgl.utils.logger import log
from pgl.utils.data import Dataloader, StreamDataset

import paddle
import paddle.nn as nn
from paddle.optimizer import Adam

from utils.config import prepare_config
from model import SkipGramModel
from datasets.dataset import TrainPairDataset, CollateFn


def data2tensor(batch_dict):
    feed_dict = {}
    for key, value in batch_dict.items():
        if isinstance(value, pgl.Graph):
            feed_dict[key] = value.tensor()
        elif isinstance(value, np.ndarray):
            feed_dict[key] = paddle.to_tensor(value)
        else:
            raise TypeError("can not convert a type of [%s] to paddle Tensor" \
                    % type(value))
    return feed_dict


def save_embedding(model, save_dir):
    save_files = os.path.join(save_dir, "embedding.txt")
    embed = model.state_dict()['embedding.weight'].numpy()
    with open(save_files, "w") as writer:
        for idx, vec in enumerate(embed):
            str_vec = ' '.join(map(str, vec))
            writer.write("%s\t%s\n" % (idx, str_vec))


def train(config, model, loader, optim):
    model.train()
    global_step = 0
    total_loss = 0.0

    start = time.time()
    for epoch in range(config.epochs):
        for step, feed_dict in enumerate(loader()):
            global_step += 1
            feed_dict = data2tensor(feed_dict)
            loss = model(feed_dict)
            loss.backward()
            optim.step()
            optim.clear_grad()

            total_loss += float(loss)
            if global_step % config.log_steps == 0:
                avg_loss = total_loss / config.log_steps
                total_loss = 0.0
                sec_per_batch = (time.time() - start) / config.log_steps
                start = time.time()
                log.info(
                    "sec/batch: %.6f | Epoch: %s | step: %s | train_loss: %.6f"
                    % (sec_per_batch, epoch, global_step, avg_loss))

        log.info("saving model in epoch %s" % (epoch))
        save_files = os.path.join(config.save_dir, "ckpt.pdparams")
        log.info("Epoch: %s | Saving model in %s" % (epoch, save_files))
        paddle.save(model.state_dict(), save_files)

    #  save_embedding(model, config.save_dir)


def main(config, ip_list_file):
    ds = TrainPairDataset(config, ip_list_file)
    loader = Dataloader(
        ds,
        batch_size=config.batch_pair_size,
        num_workers=config.num_workers,
        stream_shuffle_size=config.pair_stream_shuffle_size,
        collate_fn=CollateFn())

    model = SkipGramModel(config)

    if config.warm_start_from:
        log.info("warm start from %s" % config.warm_start_from)
        model.set_state_dict(paddle.load(config.warm_start_from))

    optim = Adam(
        learning_rate=config.lr,
        parameters=model.parameters(),
        lazy_mode=config.lazy_mode)

    log.info("starting training...")
    train(config, model, loader, optim)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GraphRec')
    parser.add_argument("--config", type=str, default="./config.yaml")
    parser.add_argument("--ip", type=str, default="./ip_list.txt")
    args = parser.parse_args()

    config = prepare_config(args.config, isCreate=True, isSave=False)
    main(config, args.ip)
