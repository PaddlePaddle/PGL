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

import numpy as np
import paddle.fluid as F
import paddle.fluid.layers as L
from pgl.utils.logger import log

from model import DeepwalkModel
from utils import build_graph
from utils import build_gen_func


def optimization(base_lr, loss, train_steps, optimizer='adam'):
    decayed_lr = L.polynomial_decay(base_lr, train_steps, 0.0001)

    if optimizer == 'sgd':
        optimizer = F.optimizer.SGD(
            decayed_lr,
            regularization=F.regularizer.L2DecayRegularizer(
                regularization_coeff=0.0025))
    elif optimizer == 'adam':
        # dont use gpu's lazy mode
        optimizer = F.optimizer.Adam(decayed_lr)
    else:
        raise ValueError

    log.info('learning rate:%f' % (base_lr))
    optimizer.minimize(loss)


def get_parallel_exe(program, loss):
    exec_strategy = F.ExecutionStrategy()
    exec_strategy.num_threads = 1  #2 for fp32 4 for fp16
    exec_strategy.use_experimental_executor = True
    exec_strategy.num_iteration_per_drop_scope = 1  #important shit

    build_strategy = F.BuildStrategy()
    build_strategy.enable_inplace = True
    build_strategy.memory_optimize = True
    build_strategy.remove_unnecessary_lock = True

    #return compiled_prog
    train_exe = F.ParallelExecutor(
        use_cuda=True,
        loss_name=loss.name,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy,
        main_program=program)
    return train_exe


def train(train_exe, exe, program, loss, node2vec_pyreader, args, train_steps):
    trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
    step = 0
    while True:
        try:
            begin_time = time.time()
            loss_val, = train_exe.run(fetch_list=[loss])
            log.info("step %s: loss %.5f speed: %.5f s/step" %
                     (step, np.mean(loss_val), time.time() - begin_time))
            step += 1
        except F.core.EOFException:
            node2vec_pyreader.reset()

        if (step == train_steps or
                step % args.steps_per_save == 0) and trainer_id == 0:

            model_save_dir = args.output_path
            model_path = os.path.join(model_save_dir, str(step))
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            F.io.save_params(exe, model_path, program)
        if step == train_steps:
            break


def main(args):
    import logging
    log.setLevel(logging.DEBUG)
    log.info("start")

    num_devices = len(F.cuda_places())
    model = DeepwalkModel(args.num_nodes, args.hidden_size, args.neg_num,
                          False, False, 1.)
    pyreader = model.pyreader
    loss = model.forward()

    train_steps = int(args.num_nodes * args.epoch / args.batch_size /
                      num_devices)
    optimization(args.lr * num_devices, loss, train_steps, args.optimizer)

    place = F.CUDAPlace(0)
    exe = F.Executor(place)
    exe.run(F.default_startup_program())

    graph = build_graph(args.num_nodes, args.edge_path)
    gen_func = build_gen_func(args, graph)

    pyreader.decorate_tensor_provider(gen_func)
    pyreader.start()

    train_prog = F.default_main_program()

    if args.warm_start_from_dir is not None:
        F.io.load_params(exe, args.warm_start_from_dir, train_prog)

    train_exe = get_parallel_exe(train_prog, loss)
    train(train_exe, exe, train_prog, loss, pyreader, args, train_steps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deepwalk')
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--neg_num", type=int, default=5)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--walk_len", type=int, default=40)
    parser.add_argument("--win_size", type=int, default=5)
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument("--num_sample_workers", type=int, default=1)
    parser.add_argument("--steps_per_save", type=int, default=3000)
    parser.add_argument("--num_nodes", type=int, default=10000)
    parser.add_argument("--edge_path", type=str, default="./graph_data")
    parser.add_argument("--walkpath_files", type=str, default=None)
    parser.add_argument("--train_files", type=str, default="./train_data")
    parser.add_argument("--warm_start_from_dir", type=str, default=None)
    parser.add_argument(
        "--neg_sample_type",
        type=str,
        default="average",
        choices=["average", "outdegree"])
    parser.add_argument(
        "--optimizer",
        type=str,
        required=False,
        choices=['adam', 'sgd'],
        default="adam")
    args = parser.parse_args()
    log.info(args)
    main(args)
