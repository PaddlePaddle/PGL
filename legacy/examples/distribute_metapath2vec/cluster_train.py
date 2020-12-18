# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
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
import argparse
import time
import os
import math
import numpy as np

import paddle.fluid as F
import paddle.fluid.layers as L
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import StrategyFactory
from pgl.utils.logger import log

from model import Metapath2vecModel
from graph import m2vGraph
from utils import load_config
from walker import multiprocess_data_generator


def optimization(base_lr, loss, train_steps, optimizer='sgd'):
    decayed_lr = L.learning_rate_scheduler.polynomial_decay(
        learning_rate=base_lr,
        decay_steps=train_steps,
        end_learning_rate=0.0001 * base_lr,
        power=1.0,
        cycle=False)
    if optimizer == 'sgd':
        optimizer = F.optimizer.SGD(decayed_lr)
    elif optimizer == 'adam':
        optimizer = F.optimizer.Adam(decayed_lr, lazy_mode=True)
    else:
        raise ValueError

    log.info('learning rate:%f' % (base_lr))
    #create the DistributeTranspiler configure
    strategy = StrategyFactory.create_async_strategy()
    optimizer = fleet.distributed_optimizer(optimizer, strategy)
    optimizer.minimize(loss)


def build_complied_prog(train_program, model_loss):
    num_threads = int(os.getenv("CPU_NUM", 10))
    trainer_id = int(os.getenv("PADDLE_TRAINER_ID", 0))
    exec_strategy = F.ExecutionStrategy()
    exec_strategy.num_threads = num_threads
    build_strategy = F.BuildStrategy()
    build_strategy.enable_inplace = True
    build_strategy.memory_optimize = False
    build_strategy.remove_unnecessary_lock = False
    if num_threads > 1:
        build_strategy.reduce_strategy = F.BuildStrategy.ReduceStrategy.Reduce

    compiled_prog = F.compiler.CompiledProgram(
        train_program).with_data_parallel(loss_name=model_loss.name)
    return compiled_prog


def train_prog(exe, program, loss, node2vec_pyreader, args, train_steps):
    trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
    step = 0
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    while True:
        try:
            begin_time = time.time()
            loss_val, = exe.run(program, fetch_list=[loss])
            log.info("step %s: loss %.5f speed: %.5f s/step" %
                     (step, np.mean(loss_val), time.time() - begin_time))
            step += 1
        except F.core.EOFException:
            node2vec_pyreader.reset()

        if step % args.steps_per_save == 0 or step == train_steps:
            save_path = args.save_path
            if trainer_id == 0:
                model_path = os.path.join(save_path, "%s" % step)
                fleet.save_persistables(exe, model_path)
        if step == train_steps:
            break


def main(args):
    log.info("start")

    worker_num = int(os.getenv("PADDLE_TRAINERS_NUM", "0"))
    num_devices = int(os.getenv("CPU_NUM", 10))

    model = Metapath2vecModel(config=args)
    pyreader = model.pyreader
    loss = model.forward()

    # init fleet
    role = role_maker.PaddleCloudRoleMaker()
    fleet.init(role)

    train_steps = math.ceil(args.num_nodes * args.epochs / args.batch_size /
                            num_devices / worker_num)
    log.info("Train step: %s" % train_steps)

    real_batch_size = args.batch_size * args.walk_len * args.win_size
    if args.optimizer == "sgd":
        args.lr *= real_batch_size
    optimization(args.lr, loss, train_steps, args.optimizer)

    # init and run server or worker
    if fleet.is_server():
        fleet.init_server(args.warm_start_from_dir)
        fleet.run_server()

    if fleet.is_worker():
        log.info("start init worker done")
        fleet.init_worker()
        #just the worker, load the sample
        log.info("init worker done")

        exe = F.Executor(F.CPUPlace())
        exe.run(fleet.startup_program)
        log.info("Startup done")

        dataset = m2vGraph(args)
        log.info("Build graph done.")

        data_generator = multiprocess_data_generator(args, dataset)

        cur_time = time.time()
        for idx, _ in enumerate(data_generator()):
            log.info("iter %s: %s s" % (idx, time.time() - cur_time))
            cur_time = time.time()
            if idx == 100:
                break

        pyreader.decorate_tensor_provider(data_generator)
        pyreader.start()

        compiled_prog = build_complied_prog(fleet.main_program, loss)
        train_prog(exe, compiled_prog, loss, pyreader, args, train_steps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='metapath2vec')
    parser.add_argument("-c", "--config", type=str, default="./config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    log.info(config)
    main(config)
