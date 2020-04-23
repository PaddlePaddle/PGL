#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
# SSL

import torch
import os
import re
import time
from random import random
from functools import reduce, partial
import numpy as np
import multiprocessing

from ogb.graphproppred import Evaluator
import paddle
import paddle.fluid as F
import paddle.fluid.layers as L
import pgl
from pgl.utils import paddle_helper
from pgl.utils.logger import log

from utils.args import print_arguments, check_cuda, prepare_logger
from utils.init import init_checkpoint, init_pretraining_params
from utils.config import Config
from optimization import optimization
from monitor.train_monitor import train_and_evaluate
from args import parser

import model as Model
from data.base_dataset import Subset, Dataset
from data.dataloader import GraphDataloader


def main(args):
    log.info('loading data')
    dataset = Dataset(args)
    args.num_class = dataset.num_tasks
    args.eval_metrics = dataset.eval_metrics
    args.task_type = dataset.task_type
    splitted_index = dataset.get_idx_split()
    train_dataset = Subset(dataset, splitted_index['train'])
    valid_dataset = Subset(dataset, splitted_index['valid'])
    test_dataset = Subset(dataset, splitted_index['test'])

    log.info("preprocess finish")
    log.info("Train Examples: %s" % len(train_dataset))
    log.info("Val Examples: %s" % len(valid_dataset))
    log.info("Test Examples: %s" % len(test_dataset))

    train_prog = F.Program()
    startup_prog = F.Program()

    if args.use_cuda:
        dev_list = F.cuda_places()
        place = dev_list[0]
        dev_count = len(dev_list)
    else:
        place = F.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
        #  dev_count = args.cpu_num

    log.info("building model")
    with F.program_guard(train_prog, startup_prog):
        with F.unique_name.guard():
            graph_model = getattr(Model, args.model_type)(args, dataset)
            train_ds = GraphDataloader(
                train_dataset,
                graph_model.graph_wrapper,
                batch_size=args.batch_size)

            num_train_examples = len(train_dataset)
            max_train_steps = args.epoch * num_train_examples // args.batch_size // dev_count
            warmup_steps = int(max_train_steps * args.warmup_proportion)

            scheduled_lr, loss_scaling = optimization(
                loss=graph_model.loss,
                warmup_steps=warmup_steps,
                num_train_steps=max_train_steps,
                learning_rate=args.learning_rate,
                train_program=train_prog,
                startup_prog=startup_prog,
                weight_decay=args.weight_decay,
                scheduler=args.lr_scheduler,
                use_fp16=False,
                use_dynamic_loss_scaling=args.use_dynamic_loss_scaling,
                init_loss_scaling=args.init_loss_scaling,
                incr_every_n_steps=args.incr_every_n_steps,
                decr_every_n_nan_or_inf=args.decr_every_n_nan_or_inf,
                incr_ratio=args.incr_ratio,
                decr_ratio=args.decr_ratio)

    test_prog = F.Program()
    with F.program_guard(test_prog, startup_prog):
        with F.unique_name.guard():
            _graph_model = getattr(Model, args.model_type)(args, dataset)

    test_prog = test_prog.clone(for_test=True)

    valid_ds = GraphDataloader(
        valid_dataset,
        graph_model.graph_wrapper,
        batch_size=args.batch_size,
        shuffle=False)
    test_ds = GraphDataloader(
        test_dataset,
        graph_model.graph_wrapper,
        batch_size=args.batch_size,
        shuffle=False)

    exe = F.Executor(place)
    exe.run(startup_prog)
    for init in graph_model.init_vars:
        init(place)
    for init in _graph_model.init_vars:
        init(place)

    if args.init_pretraining_params is not None:
        init_pretraining_params(
            exe, args.init_pretraining_params, main_program=startup_prog)

    nccl2_num_trainers = 1
    nccl2_trainer_id = 0
    if dev_count > 1:

        exec_strategy = F.ExecutionStrategy()
        exec_strategy.num_threads = dev_count

        train_exe = F.ParallelExecutor(
            use_cuda=args.use_cuda,
            loss_name=graph_model.loss.name,
            exec_strategy=exec_strategy,
            main_program=train_prog,
            num_trainers=nccl2_num_trainers,
            trainer_id=nccl2_trainer_id)

        test_exe = exe
    else:
        train_exe, test_exe = exe, exe

    evaluator = Evaluator(args.dataset_name)

    train_and_evaluate(
        exe=exe,
        train_exe=train_exe,
        valid_exe=test_exe,
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
        train_prog=train_prog,
        valid_prog=test_prog,
        args=args,
        dev_count=dev_count,
        evaluator=evaluator,
        model=graph_model)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.config is not None:
        args = Config(args.config, isCreate=True, isSave=True)

    log.info(args)

    main(args)
