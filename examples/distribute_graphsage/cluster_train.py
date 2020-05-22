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
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from pgl.utils.logger import log

from model import GraphsageModel
from utils import load_config
import reader


def init_role():
    # reset the place according to role of parameter server
    training_role = os.getenv("TRAINING_ROLE", "TRAINER")
    paddle_role = role_maker.Role.WORKER
    place = F.CPUPlace()
    if training_role == "PSERVER":
        paddle_role = role_maker.Role.SERVER

    # set the fleet runtime environment according to configure
    ports = os.getenv("PADDLE_PORT", "6174").split(",")
    pserver_ips = os.getenv("PADDLE_PSERVERS").split(",")  # ip,ip...
    eplist = []
    if len(ports) > 1:
        # local debug mode, multi port
        for port in ports:
            eplist.append(':'.join([pserver_ips[0], port]))
    else:
        # distributed mode, multi ip
        for ip in pserver_ips:
            eplist.append(':'.join([ip, ports[0]]))

    pserver_endpoints = eplist  # ip:port,ip:port...
    worker_num = int(os.getenv("PADDLE_TRAINERS_NUM", "0"))
    trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
    role = role_maker.UserDefinedRoleMaker(
        current_id=trainer_id,
        role=paddle_role,
        worker_num=worker_num,
        server_endpoints=pserver_endpoints)
    fleet.init(role)


def optimization(base_lr, loss, optimizer='adam'):
    if optimizer == 'sgd':
        optimizer = F.optimizer.SGD(base_lr)
    elif optimizer == 'adam':
        optimizer = F.optimizer.Adam(base_lr, lazy_mode=True)
    else:
        raise ValueError

    log.info('learning rate:%f' % (base_lr))
    #create the DistributeTranspiler configure
    config = DistributeTranspilerConfig()
    config.sync_mode = False
    #config.runtime_split_send_recv = False

    config.slice_var_up = False
    #create the distributed optimizer
    optimizer = fleet.distributed_optimizer(optimizer, config)
    optimizer.minimize(loss)


def build_complied_prog(train_program, model_loss):
    num_threads = int(os.getenv("CPU_NUM", 10))
    trainer_id = int(os.getenv("PADDLE_TRAINER_ID", 0))
    exec_strategy = F.ExecutionStrategy()
    exec_strategy.num_threads = num_threads
    #exec_strategy.use_experimental_executor = True
    build_strategy = F.BuildStrategy()
    build_strategy.enable_inplace = True
    #build_strategy.memory_optimize = True
    build_strategy.memory_optimize = False
    build_strategy.remove_unnecessary_lock = False
    if num_threads > 1:
        build_strategy.reduce_strategy = F.BuildStrategy.ReduceStrategy.Reduce

    compiled_prog = F.compiler.CompiledProgram(
        train_program).with_data_parallel(loss_name=model_loss.name)
    return compiled_prog


def fake_py_reader(data_iter, num):
    def fake_iter():
        queue = []
        for idx, data in enumerate(data_iter()):
            queue.append(data)
            if len(queue) == num:
                yield queue
                queue = []
        if len(queue) > 0:
            while len(queue) < num:
                queue.append(queue[-1])
            yield queue
    return fake_iter

def train_prog(exe, program, model, pyreader, args):
    trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
    start = time.time()
    batch = 0
    total_loss = 0.
    total_acc = 0.
    total_sample = 0
    for epoch_idx in range(args.num_epoch):
        for step, batch_feed_dict in enumerate(pyreader()):
            try:
                cpu_time = time.time()
                batch += 1
                batch_loss, batch_acc  = exe.run(
                    program,
                    feed=batch_feed_dict,
                    fetch_list=[model.loss, model.acc])

                end = time.time()
                if batch % args.log_per_step == 0:
                    log.info(
                        "Batch %s Loss %s Acc %s \t Speed(per batch) %.5lf/%.5lf sec"
                        % (batch, np.mean(batch_loss), np.mean(batch_acc), (end - start) /batch, (end - cpu_time)))

                if step % args.steps_per_save == 0:
                    save_path = args.save_path
                    if trainer_id == 0:
                        model_path = os.path.join(save_path, "%s" % step)
                        fleet.save_persistables(exe, model_path)
            except Exception as e:
                log.info("Pyreader train error")
                log.exception(e)

def main(args):
    log.info("start")

    worker_num = int(os.getenv("PADDLE_TRAINERS_NUM", "0"))
    num_devices = int(os.getenv("CPU_NUM", 10))

    model = GraphsageModel(args)
    loss = model.forward()
    train_iter = reader.get_iter(args, model.graph_wrapper, 'train')
    pyreader = fake_py_reader(train_iter, num_devices)

    # init fleet
    init_role()

    optimization(args.lr, loss, args.optimizer)

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

        compiled_prog = build_complied_prog(fleet.main_program, loss)
        train_prog(exe, compiled_prog, model, pyreader, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='metapath2vec')
    parser.add_argument("-c", "--config", type=str, default="./config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    log.info(config)
    main(config)

