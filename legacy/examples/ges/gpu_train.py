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
""" gpu_train
"""
import argparse
import time
import os
import glob

import numpy as np
import paddle.fluid as F
import paddle.fluid.layers as L
from pgl.utils.logger import log
from pgl.graph import Graph
from pgl.sample import graph_alias_sample_table
from pgl import data_loader

import mp_reader
from reader import GESReader
from model import GESModel


def get_file_list(path):
    """get_file_list
    """
    filelist = []
    if os.path.isfile(path):
        filelist = [path]
    elif os.path.isdir(path):
        filelist = [
            os.path.join(dp, f)
            for dp, dn, filenames in os.walk(path) for f in filenames
        ]
    else:
        raise ValueError(path + " not supported")
    return filelist


def build_graph(num_nodes, edge_path, output_path, undigraph=True):
    """ build_graph
    """
    edge_file = os.path.join(output_path, "edge.npy")
    edge_weight_file = os.path.join(output_path, "edge_weight.npy")
    alias_file = os.path.join(output_path, "alias.npy")
    events_file = os.path.join(output_path, "events.npy")
    if os.path.isfile(edge_file):
        edges = np.load(edge_file)
        edge_feat = dict()
        if os.path.isfile(edge_weight_file):
            log.info("Loading weight from cache")
            edge_feat["weight"] = np.load(edge_weight_file, allow_pickle=True)
        node_feat = dict()
        if os.path.isfile(alias_file):
            log.info("Loading alias from cache")
            node_feat["alias"] = np.load(alias_file, allow_pickle=True)
        if os.path.isfile(events_file):
            log.info("Loading events from cache")
            node_feat["events"] = np.load(events_file, allow_pickle=True)
    else:
        filelist = get_file_list(edge_path)
        edges, edge_weight = [], []
        log.info("Reading edge files")
        for name in filelist:
            with open(name) as inf:
                for line in inf:
                    slots = line.strip("\n").split()
                    edges.append([slots[0], slots[1]])
                    if len(slots) > 2:
                        edge_weight.append(slots[2])
        edges = np.array(edges, dtype="int64")
        assert num_nodes > edges.max(
        ), "Node id in any edges should be smaller then num_nodes!"

        log.info("Read edge files done.")
        edge_feat = dict()
        node_feat = dict()
        if len(edge_weight) == len(edges):
            edge_feat["weight"] = np.array(edge_weight, dtype="float32")

    if undigraph is True:
        edges = np.concatenate([edges, edges[:, [1, 0]]], 0)
        if "weight" in edge_feat:
            edge_feat["weight"] = np.concatenate(
                [edge_feat["weight"], edge_feat["weight"]],
                0).astype("float64")

    graph = Graph(num_nodes, edges, node_feat, edge_feat=edge_feat)
    log.info("Build graph done")
    graph.outdegree()
    log.info("Build graph index done")
    if "weight" in graph.edge_feat and "alias" not in graph.node_feat and "events" not in graph.node_feat:
        graph.node_feat["alias"], graph.node_feat[
            "events"] = graph_alias_sample_table(graph, "weight")
        log.info(
            "Build graph alias sample table done, and saving alias & evnets cache"
        )
        np.save(alias_file, graph.node_feat["alias"])
        np.save(events_file, graph.node_feat["events"])
    return graph


def optimization(base_lr, loss, train_steps, optimizer='adam'):
    """ optimization
    """
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


def build_gen_func(args, graph, node_feat):
    """ build_gen_func
    """
    num_sample_workers = args.num_sample_workers

    if args.walkpath_files is None:
        walkpath_files = [None for _ in range(num_sample_workers)]
    else:
        files = get_file_list(args.walkpath_files)
        walkpath_files = [[] for i in range(num_sample_workers)]
        for idx, f in enumerate(files):
            walkpath_files[idx % num_sample_workers].append(f)

    if args.train_files is None:
        train_files = [None for _ in range(num_sample_workers)]
    else:
        files = get_file_list(args.train_files)
        train_files = [[] for i in range(num_sample_workers)]
        for idx, f in enumerate(files):
            train_files[idx % num_sample_workers].append(f)

    gen_func_pool = [
        GESReader(
            graph,
            node_feat,
            batch_size=args.batch_size,
            walk_len=args.walk_len,
            win_size=args.win_size,
            neg_num=args.neg_num,
            neg_sample_type=args.neg_sample_type,
            walkpath_files=walkpath_files[i],
            train_files=train_files[i]) for i in range(num_sample_workers)
    ]
    if num_sample_workers == 1:
        gen_func = gen_func_pool[0]
    else:
        gen_func = mp_reader.multiprocess_reader(
            gen_func_pool, use_pipe=True, queue_size=100)
    return gen_func


def get_parallel_exe(program, loss):
    """ get_parallel_exe
    """
    exec_strategy = F.ExecutionStrategy()
    exec_strategy.num_threads = 1  #2 for fp32 4 for fp16
    exec_strategy.use_experimental_executor = True
    exec_strategy.num_iteration_per_drop_scope = 10  #important shit

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
    """ train
    """
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

        if (step % args.steps_per_save == 0 or
                step == train_steps) and trainer_id == 0:

            model_save_dir = args.output_path
            model_path = os.path.join(model_save_dir, str(step))
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            F.io.save_params(exe, model_path, program)

        if step == train_steps:
            break


def test_gen_speed(gen_func):
    """ test_gen_speed
    """
    cur_time = time.time()
    for idx, _ in enumerate(gen_func()):
        log.info("iter %s: %s s" % (idx, time.time() - cur_time))
        cur_time = time.time()
        if idx == 100:
            break


def main(args):
    """ main
    """
    import logging
    log.setLevel(logging.DEBUG)
    log.info("start")

    if args.dataset is not None:
        if args.dataset == "BlogCatalog":
            graph = data_loader.BlogCatalogDataset().graph
        else:
            raise ValueError(args.dataset + " dataset doesn't exists")
        log.info("Load buildin BlogCatalog dataset done.")
        node_feat = np.expand_dims(graph.node_feat["group_id"].argmax(-1),
                                   -1) + graph.num_nodes
        args.num_nodes = graph.num_nodes
        args.num_embedding = graph.num_nodes + graph.node_feat[
            "group_id"].shape[-1]
    else:
        graph = build_graph(args.num_nodes, args.edge_path, args.output_path)
        node_feat = np.load(args.node_feat_npy)

    model = GESModel(args.num_embedding, node_feat.shape[1] + 1,
                     args.hidden_size, args.neg_num, False, 2)
    pyreader = model.pyreader
    loss = model.forward()
    num_devices = len(F.cuda_places())

    train_steps = int(args.num_nodes * args.epoch / args.batch_size /
                      num_devices)
    log.info("Train steps: %s" % train_steps)
    optimization(args.lr * num_devices, loss, train_steps, args.optimizer)

    place = F.CUDAPlace(0)
    exe = F.Executor(place)
    exe.run(F.default_startup_program())

    gen_func = build_gen_func(args, graph, node_feat)

    pyreader.decorate_tensor_provider(gen_func)
    pyreader.start()
    train_prog = F.default_main_program()
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
    parser.add_argument("--num_embedding", type=int, default=10000)
    parser.add_argument("--edge_path", type=str, default="./graph_data")
    parser.add_argument("--walkpath_files", type=str, default=None)
    parser.add_argument("--train_files", type=str, default="./train_data")
    parser.add_argument("--node_feat_npy", type=str, default="./feat.npy")
    parser.add_argument("--dataset", type=str, default=None)
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
