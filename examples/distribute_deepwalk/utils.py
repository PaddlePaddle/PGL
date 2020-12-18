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
"""
    Utils file.
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os
import time

import numpy as np
from pgl.utils.logger import log
from pgl.graph import Graph
from pgl.sample import graph_alias_sample_table

from reader import DeepwalkReader
import mp_reader


def get_file_list(path):
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


def build_graph(num_nodes, edge_path):
    filelist = []
    if os.path.isfile(edge_path):
        filelist = [edge_path]
    elif os.path.isdir(edge_path):
        filelist = [
            os.path.join(dp, f)
            for dp, dn, filenames in os.walk(edge_path) for f in filenames
        ]
    else:
        raise ValueError(edge_path + " not supported")
    edges, edge_weight = [], []
    for name in filelist:
        with open(name) as inf:
            for line in inf:
                slots = line.strip("\n").split()
                edges.append([slots[0], slots[1]])
                edges.append([slots[1], slots[0]])
                if len(slots) > 2:
                    edge_weight.extend([float(slots[2]), float(slots[2])])
    edges = np.array(edges, dtype="int64")
    assert num_nodes > edges.max(
    ), "Node id in any edges should be smaller then num_nodes!"

    edge_feat = dict()
    if len(edge_weight) == len(edges):
        edge_feat["weight"] = np.array(edge_weight)

    graph = Graph(num_nodes, edges, edge_feat=edge_feat)
    log.info("Build graph done")

    graph.outdegree()

    del edges, edge_feat

    log.info("Build graph index done")
    if "weight" in graph.edge_feat:
        graph.node_feat["alias"], graph.node_feat[
            "events"] = graph_alias_sample_table(graph, "weight")
        log.info("Build graph alias sample table done")
    return graph


def build_fake_graph(num_nodes):
    class FakeGraph():
        pass

    graph = FakeGraph()
    graph.num_nodes = num_nodes
    return graph


def build_random_graph(num_nodes):
    edges = np.random.randint(0, num_nodes, [4*num_nodes, 2])
    graph = pgl.graph.Graph(num_nodes, edges)
    graph.indegree()
    graph.outdegree()
    return graph


def build_gen_func(args, graph):
    num_sample_workers = args.num_sample_workers

    if args.walkpath_files is None or args.walkpath_files == "None":
        walkpath_files = [None for _ in range(num_sample_workers)]
    else:
        files = get_file_list(args.walkpath_files)
        walkpath_files = [[] for i in range(num_sample_workers)]
        for idx, f in enumerate(files):
            walkpath_files[idx % num_sample_workers].append(f)

    if args.train_files is None or args.train_files == "None":
        train_files = [None for _ in range(num_sample_workers)]
    else:
        files = get_file_list(args.train_files)
        train_files = [[] for i in range(num_sample_workers)]
        for idx, f in enumerate(files):
            train_files[idx % num_sample_workers].append(f)

    gen_func_pool = [
        DeepwalkReader(
            graph,
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


def test_gen_speed(gen_func):
    cur_time = time.time()
    for idx, _ in enumerate(gen_func()):
        log.info("iter %s: %s s" % (idx, time.time() - cur_time))
        cur_time = time.time()
        if idx == 100:
            break
