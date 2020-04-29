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
"""doc
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import time
import io
import os
import numpy as np
import random

from pgl.utils.logger import log
from pgl.sample import metapath_randomwalk
from pgl.graph_kernel import skip_gram_gen_pair
from pgl.graph_kernel import alias_sample_build_table

from utils import load_config
from graph import m2vGraph
import mp_reader


class NodeGenerator(object):
    """Node generator"""

    def __init__(self, config, graph):
        self.config = config
        self.graph = graph

        self.batch_size = self.config.batch_size
        self.shuffle = self.config.node_shuffle
        self.node_files = self.config.node_files
        self.first_node_type = self.config.first_node_type
        self.walk_mode = self.config.walk_mode

    def __call__(self):
        if self.walk_mode == "m2v":
            generator = self.m2v_node_generate
            log.info("node gen mode is : %s" % (self.walk_mode))
        elif self.walk_mode == "multi_m2v":
            generator = self.multi_m2v_node_generate
            log.info("node gen mode is : %s" % (self.walk_mode))
        elif self.walk_mode == "files":
            generator = self.files_node_generate
            log.info("node gen mode is : %s" % (self.walk_mode))
        else:
            generator = self.m2v_node_generate
            log.info("node gen mode is : %s" % (self.walk_mode))

        while True:
            for nodes in generator():
                yield nodes

    def m2v_node_generate(self):
        """m2v_node_generate"""
        for nodes in self.graph.node_batch_iter(
                batch_size=self.batch_size,
                n_type=self.first_node_type,
                shuffle=self.shuffle):
            yield nodes

    def multi_m2v_node_generate(self):
        """multi_m2v_node_generate"""
        n_type_list = self.first_node_type.split(';')
        num_n_type = len(n_type_list)
        node_types = np.unique(self.graph.node_types).tolist()

        node_generators = {}
        for n_type in node_types:
            node_generators[n_type] = \
                    self.graph.node_batch_iter(self.batch_size, n_type=n_type)

        cc = 0
        while True:
            idx = cc % num_n_type
            n_type = n_type_list[idx]
            try:
                nodes = next(node_generators[n_type])
            except StopIteration as e:
                log.info("node type of %s iteration finished in one epoch" %
                         (n_type))
                node_generators[n_type] = \
                        self.graph.node_batch_iter(self.batch_size, n_type=n_type)
                break
            yield (nodes, idx)
            cc += 1

    def files_node_generate(self):
        """files_node_generate"""
        nodes = []
        for filename in self.node_files:
            with io.open(filename) as inf:
                for line in inf:
                    node = int(line.strip('\n\t'))
                    nodes.append(node)
                    if len(nodes) == self.batch_size:
                        yield nodes
                        nodes = []
        if len(nodes):
            yield nodes


class WalkGenerator(object):
    """Walk generator"""

    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset
        self.graph = self.dataset.graph
        self.walk_mode = self.config.walk_mode
        self.node_generator = NodeGenerator(self.config, self.graph)

        if self.walk_mode == "multi_m2v":
            num_path = len(self.config.meta_path.split(';'))
            num_first_node_type = len(self.config.first_node_type.split(';'))
            assert num_first_node_type == num_path, \
                "In [multi_m2v] walk_mode, the number of metapath should be the same \
                as the number of first_node_type"

            assert num_path > 1, "In [multi_m2v] walk_mode, the number of metapath\
                    should be greater than 1"

    def __call__(self):
        np.random.seed(os.getpid())
        if self.walk_mode == "m2v":
            walk_generator = self.m2v_walk
            log.info("walk mode is : %s" % (self.walk_mode))
        elif self.walk_mode == "multi_m2v":
            walk_generator = self.multi_m2v_walk
            log.info("walk mode is : %s" % (self.walk_mode))
        else:
            raise ValueError("walk_mode [%s] is not matched" % self.walk_mode)

        for walks in walk_generator():
            yield walks

    def m2v_walk(self):
        """Metapath2vec walker"""
        for nodes in self.node_generator():
            walks = metapath_randomwalk(
                self.graph, nodes, self.config.meta_path, self.config.walk_len)

            yield walks

    def multi_m2v_walk(self):
        """Multi metapath2vec walker"""
        meta_paths = self.config.meta_path.split(';')
        for nodes, idx in self.node_generator():
            walks = metapath_randomwalk(self.graph, nodes, meta_paths[idx],
                                        self.config.walk_len)
            yield walks


class DataGenerator(object):
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset
        self.graph = self.dataset.graph
        self.walk_generator = WalkGenerator(self.config, self.dataset)

    def __call__(self):
        generator = self.pair_generate

        for src, pos, negs in generator():
            dst = np.concatenate([pos, negs], 1)
            yield src, dst

    def pair_generate(self):
        for walks in self.walk_generator():
            try:
                src_list, pos_list = [], []
                for walk in walks:
                    s, p = skip_gram_gen_pair(walk, self.config.win_size)
                    src_list.append(s), pos_list.append(p)
                src = [s for x in src_list for s in x]
                pos = [s for x in pos_list for s in x]

                if len(src) == 0:
                    continue

                negs = self.negative_sample(
                    src,
                    pos,
                    neg_num=self.config.neg_num,
                    neg_sample_type=self.config.neg_sample_type)

                src = np.array(src, dtype=np.int64).reshape(-1, 1, 1)
                pos = np.array(pos, dtype=np.int64).reshape(-1, 1, 1)

                yield src, pos, negs

            except Exception as e:
                log.exception(e)

    def negative_sample(self, src, pos, neg_num, neg_sample_type):
        if neg_sample_type == "average":
            neg_sample_size = [len(pos), neg_num, 1]
            negs = np.random.randint(
                low=0, high=self.graph.num_nodes, size=neg_sample_size)

        elif neg_sample_type == "m2v_plus":
            negs = []
            for s in src:
                neg = self.graph.sample_nodes(
                    sample_num=neg_num, n_type=self.graph.node_types[s])
                negs.append(neg)
            negs = np.vstack(negs).reshape(-1, neg_num, 1)

        else:  # equal to "average"
            neg_sample_size = [len(pos), neg_num, 1]
            negs = np.random.randint(
                low=0, high=self.graph.num_nodes, size=neg_sample_size)

        negs = negs.astype(np.int64)

        return negs


def multiprocess_data_generator(config, dataset):
    """Multiprocess data generator.
    """
    if config.num_sample_workers == 1:
        data_generator = DataGenerator(config, dataset)
    else:
        pool = [
            DataGenerator(config, dataset)
            for i in range(config.num_sample_workers)
        ]
        data_generator = mp_reader.multiprocess_reader(
            pool, use_pipe=True, queue_size=100)

    return data_generator


if __name__ == "__main__":
    config_file = "./config.yaml"
    config = load_config(config_file)
    dataset = m2vGraph(config)
    data_generator = multiprocess_data_generator(config, dataset)
    start = time.time()
    cc = 0
    for src, dst in data_generator():
        log.info(src.shape)

        log.info("time: %.6f" % (time.time() - start))
        start = time.time()
        cc += 1
        if cc == 100:
            break
