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
from multiprocessing import Process

import numpy as np
import paddle.fluid as F
import paddle.fluid.layers as L
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import StrategyFactory
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
import pgl
from pgl.utils.logger import log
from pgl import data_loader

from reader import DeepwalkReader
from model import DeepwalkModel
from utils import get_file_list
from utils import build_graph
from utils import build_random_graph
from utils import build_fake_graph
from utils import build_gen_func


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
    #create the distributed optimizer

    #strategy = StrategyFactory.create_sync_strategy()
    strategy = StrategyFactory.create_async_strategy()
    #strategy = StrategyFactory.create_half_async_strategy()
    #strategy = StrategyFactory.create_geo_strategy()
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
        train_program).with_data_parallel(
            loss_name=model_loss.name)
    return compiled_prog


def train_prog(exe, program, loss, pyreader, args, train_steps):
    trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
    step = 0
    while True:
        try:
            begin_time = time.time()
            loss_val, = exe.run(program, fetch_list=[loss])
            log.info("step %s: loss %.5f speed: %.5f s/step" %
                     (step, np.mean(loss_val), time.time() - begin_time))
            step += 1
        except F.core.EOFException:
            pyreader.reset()

        if step % args.steps_per_save == 0 or step == train_steps:
            if trainer_id == 0 or args.is_distributed:
                model_save_dir = args.save_path
                model_path = os.path.join(model_save_dir, str(step))
                if not os.path.exists(model_save_dir):
                    os.makedirs(model_save_dir)
                fleet.save_persistables(exe, model_path)

        if step == train_steps:
            break


def test(args):
    graph = build_graph(args.num_nodes, args.edge_path)
    gen_func = build_gen_func(args, graph)

    start = time.time()
    num = 10
    for idx, _ in enumerate(gen_func()):
        if idx % num == num - 1:
            log.info("%s" % (1.0 * (time.time() - start) / num))
            start = time.time()


def walk(args):
    graph = build_graph(args.num_nodes, args.edge_path)
    num_sample_workers = args.num_sample_workers

    if args.train_files is None or args.train_files == "None":
        log.info("Walking from graph...")
        train_files = [None for _ in range(num_sample_workers)]
    else:
        log.info("Walking from train_data...")
        files = get_file_list(args.train_files)
        train_files = [[] for i in range(num_sample_workers)]
        for idx, f in enumerate(files):
            train_files[idx % num_sample_workers].append(f)

    def walk_to_file(walk_gen, filename, max_num):
        with open(filename, "w") as outf:
            num = 0
            for walks in walk_gen:
                for walk in walks:
                    outf.write("%s\n" % "\t".join([str(i) for i in walk]))
                    num += 1
                    if num % 1000 == 0:
                        log.info("Total: %s, %s walkpath is saved. " %
                                 (max_num, num))
                    if num == max_num:
                        return

    m_args = [(DeepwalkReader(
        graph,
        batch_size=args.batch_size,
        walk_len=args.walk_len,
        win_size=args.win_size,
        neg_num=args.neg_num,
        neg_sample_type=args.neg_sample_type,
        walkpath_files=None,
        train_files=train_files[i]).walk_generator(),
               "%s/%s" % (args.walkpath_files, i),
               args.epoch * args.num_nodes // args.num_sample_workers)
              for i in range(num_sample_workers)]
    ps = []
    for i in range(num_sample_workers):
        p = Process(target=walk_to_file, args=m_args[i])
        p.start()
        ps.append(p)
    for i in range(num_sample_workers):
        ps[i].join()


def test_data_pipeline(gen_func):
    import time
    tmp = time.time()
    for idx, data in enumerate(gen_func()):
        print(time.time() - tmp)
        tmp = time.time()
        if idx == 100:
            break


def train(args):
    import logging
    log.setLevel(logging.DEBUG)
    log.info("start")

    worker_num = int(os.getenv("PADDLE_TRAINERS_NUM", "0"))
    num_devices = int(os.getenv("CPU_NUM", 10))

    model = DeepwalkModel(args.num_nodes, args.hidden_size, args.neg_num,
                          args.is_sparse, args.is_distributed, 1.)
    pyreader = model.pyreader
    loss = model.forward()

    # init fleet
    role = role_maker.PaddleCloudRoleMaker()
    fleet.init(role)

    train_steps = math.ceil(1. * args.num_nodes * args.epoch /
                            args.batch_size / num_devices / worker_num)
    log.info("Train step: %s" % train_steps)

    if args.optimizer == "sgd":
        args.lr *= args.batch_size * args.walk_len * args.win_size
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

        if args.dataset is not None:
            # load graph from built-in dataset
            if args.dataset == "BlogCatalog":    
                graph = data_loader.BlogCatalogDataset().graph    
            elif args.dataset == "ArXiv":    
                graph = data_loader.ArXivDataset().graph    
            else:    
                raise ValueError(args.dataset + " dataset doesn't exists")    
            log.info("Load buildin BlogCatalog dataset done.")    
        elif args.walkpath_files is None or args.walkpath_files == "None":    
            # build graph from edges file.
            graph = build_graph(args.num_nodes, args.edge_path)    
            log.info("Load graph from '%s' done." % args.edge_path)    
        else:
            # build a random graph for test
            graph = build_random_graph(args.num_nodes)

        # bind gen
        gen_func = build_gen_func(args, graph)
        test_data_pipeline(gen_func)

        pyreader.decorate_tensor_provider(gen_func)
        pyreader.start()

        compiled_prog = build_complied_prog(fleet.main_program, loss)
        train_prog(exe, compiled_prog, loss, pyreader, args, train_steps)
        fleet.stop_worker()



if __name__ == '__main__':

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description='Deepwalk')
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=64,
        help="Hidden size of the embedding.")
    parser.add_argument(
        "--lr", type=float, default=0.025, help="Learning rate.")
    parser.add_argument(
        "--neg_num", type=int, default=5, help="Number of negative samples.")
    parser.add_argument(
        "--epoch", type=int, default=1000, help="Number of training epoch.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Numbert of walk paths in a batch.")
    parser.add_argument(
        "--walk_len", type=int, default=40, help="Length of a walk path.")
    parser.add_argument(
        "--win_size", type=int, default=5, help="Window size in skip-gram.")
    parser.add_argument(
        "--save_path",
        type=str,
        default="model_path",
        help="Output path for saving model.")
    parser.add_argument(
        "--num_sample_workers",
        type=int,
        default=1,
        help="Number of sampling workers.")
    parser.add_argument(
        "--steps_per_save",
        type=int,
        default=100,
        help="Steps for model saveing.")
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=100000,
        help="Number of nodes in graph.")
    parser.add_argument("--edge_path", type=str, default="./graph_data")
    parser.add_argument("--train_files", type=str, default=None)
    parser.add_argument("--walkpath_files", type=str, default=None)
    parser.add_argument("--is_distributed", type=str2bool, default=False)
    parser.add_argument("--is_sparse", type=str2bool, default=True)
    parser.add_argument("--warm_start_from_dir", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="BlogCatalog")
    parser.add_argument(
        "--neg_sample_type",
        type=str,
        default="average",
        choices=["average", "outdegree"])
    parser.add_argument(
        "--mode",
        type=str,
        required=False,
        choices=['train', 'walk'],
        default="train")
    parser.add_argument(
        "--optimizer",
        type=str,
        required=False,
        choices=['adam', 'sgd'],
        default="adam")
    args = parser.parse_args()
    log.info(args)
    if args.mode == "train":
        train(args)
    elif args.mode == "walk":
        walk(args)
