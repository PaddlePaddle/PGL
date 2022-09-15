# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved
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
"""doc
"""

import os
import sys
import time
import tqdm
import yaml
import argparse
import numpy as np

import pgl
from pgl.utils.logger import log

import paddle
import paddle.nn as nn
from paddle.optimizer import Adam
import paddle.distributed.fleet as fleet

from utils.config import prepare_config
from utils.logger import log_to_file
import models as M
from datasets.dist_dataloader import DistCPUDataloader
import datasets.dataset as DS

paddle.set_device("cpu")
paddle.enable_static()
fleet.init()
time.sleep(3)
START = time.time()


def main(config, ip_list_file):
    config.embed_type = "SparseEmbedding"
    model = getattr(M, config.model_type)(config, mode="distcpu")
    feed_dict, py_reader = model.get_static_input()
    pred = model(feed_dict)
    loss = model.loss(pred)

    optimizer = paddle.optimizer.SGD(config.lr)
    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.a_sync = True
    optimizer = fleet.distributed_optimizer(optimizer, dist_strategy)
    optimizer.minimize(loss)

    if fleet.is_server():
        if config.warm_start_from:
            log.info("warm start from %s" % config.warm_start_from)
            fleet.init_server(config.warm_start_from)
        else:
            fleet.init_server()
        fleet.run_server()

    if fleet.is_worker():
        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        exe.run(paddle.static.default_startup_program())

        ds = getattr(DS, config.dataset_type)(config,
                                              ip_list_file,
                                              mode="distcpu_train")
        loader = DistCPUDataloader(
            ds,
            batch_size=config.batch_pair_size,
            num_workers=config.num_workers,
            stream_shuffle_size=config.pair_stream_shuffle_size,
            collate_fn=getattr(DS, config.collatefn)(config, mode="distcpu"))

        fleet.init_worker()
        py_reader.set_batch_generator(lambda: loader)
        fleet.barrier_worker()
        log.info("begin training in worker [%s]" % fleet.worker_index())
        train(config, exe,
              paddle.static.default_main_program(), py_reader, loss)
        log.info("training finished in worker [%s], waiting other workers..." \
                % (fleet.worker_index()))
        log.info("training time is %s" % (time.time() - START))
        fleet.barrier_worker()
        log.info("stopping workers...")
        fleet.stop_worker()
        log.info("successfully stopping worker [%s]" % fleet.worker_index())


def train(config, exe, program, reader, loss):
    total_loss = 0.0
    global_step = 0
    reader.start()
    start = time.time()
    try:
        while True:
            global_step += 1
            t_loss, = exe.run(program, fetch_list=[loss.name])
            t_loss = t_loss.mean()

            total_loss += t_loss
            if global_step % config.log_steps == 0:
                avg_loss = total_loss / config.log_steps
                total_loss = 0.0
                sec_per_batch = (time.time() - start) / config.log_steps
                start = time.time()
                log.info("sec/batch: %.6f | step: %s | train_loss: %.6f" %
                         (sec_per_batch, global_step, avg_loss))

            if fleet.is_first_worker(
            ) and global_step % config.save_steps == 0:
                log.info("saving model on step %s to %s" %
                         (global_step, config.save_dir))
                fleet.save_persistables(exe, config.save_dir,
                                        paddle.static.default_main_program())
    except paddle.framework.core.EOFException:
        reader.reset()

    if fleet.is_first_worker():
        log.info("saving model to %s after training" % (config.save_dir))
        fleet.save_persistables(exe, config.save_dir,
                                paddle.static.default_main_program())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GraphRec')
    parser.add_argument("--config", type=str, default="./config.yaml")
    parser.add_argument("--ip", type=str, default="./ip_list.txt")
    parser.add_argument("--task_name", type=str, default="graph_rec")
    args = parser.parse_args()

    config = prepare_config(
        args.config,
        isCreate=True,
        isSave=True,
        worker_index=fleet.worker_index())
    log_to_file(log, config.log_dir)

    log.info(
        "========================================================================="
    )
    for key, value in config.items():
        log.info("%s: %s" % (key, value))
    log.info(
        "========================================================================="
    )

    main(config, args.ip)
    end = time.time()
    log.info("finished training with %s seconds" % (end - START))
