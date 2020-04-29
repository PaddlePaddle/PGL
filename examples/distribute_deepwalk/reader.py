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
    Reader file.
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import time
import io
import os

import numpy as np
import paddle
from pgl.utils.logger import log
from pgl.sample import node2vec_sample
from pgl.sample import deepwalk_sample
from pgl.sample import alias_sample
from pgl.graph_kernel import skip_gram_gen_pair
from pgl.graph_kernel import alias_sample_build_table
from pgl.utils import mp_reader


class DeepwalkReader(object):
    def __init__(self,
                 graph,
                 batch_size=512,
                 walk_len=40,
                 win_size=5,
                 neg_num=5,
                 train_files=None,
                 walkpath_files=None,
                 neg_sample_type="average"):
        """
        Args:
            walkpath_files: if is not None, read walk path from walkpath_files
        """
        self.graph = graph
        self.batch_size = batch_size
        self.walk_len = walk_len
        self.win_size = win_size
        self.neg_num = neg_num
        self.train_files = train_files
        self.walkpath_files = walkpath_files
        self.neg_sample_type = neg_sample_type

    def walk_from_files(self):
        bucket = []
        while True:
            for filename in self.walkpath_files:
                with io.open(filename) as inf:
                    for line in inf:
                        #walk = [hash_map[x] for x in line.strip('\n\t').split('\t')]
                        walk = [int(x) for x in line.strip('\n\t').split('\t')]
                        bucket.append(walk)
                        if len(bucket) == self.batch_size:
                            yield bucket
                            bucket = []
            if len(bucket):
                yield bucket

    def walk_from_graph(self):
        def node_generator():
            if self.train_files is None:
                while True:
                    for nodes in self.graph.node_batch_iter(self.batch_size):
                        yield nodes
            else:
                nodes = []
                while True:
                    for filename in self.train_files:
                        with io.open(filename) as inf:
                            for line in inf:
                                node = int(line.strip('\n\t'))
                                nodes.append(node)
                                if len(nodes) == self.batch_size:
                                    yield nodes
                                    nodes = []
                if len(nodes):
                    yield nodes

        if "alias" in self.graph.node_feat and "events" in self.graph.node_feat:
            log.info("Deepwalk using alias sample")
        for nodes in node_generator():
            if "alias" in self.graph.node_feat and "events" in self.graph.node_feat:
                walks = deepwalk_sample(self.graph, nodes, self.walk_len,
                                        "alias", "events")
            else:
                walks = deepwalk_sample(self.graph, nodes, self.walk_len)
            yield walks

    def walk_generator(self):
        if self.walkpath_files is not None:
            for i in self.walk_from_files():
                yield i
        else:
            for i in self.walk_from_graph():
                yield i

    def __call__(self):
        np.random.seed(os.getpid())
        if self.neg_sample_type == "outdegree":
            outdegree = self.graph.outdegree()
            distribution = 1. * outdegree / outdegree.sum()
            alias, events = alias_sample_build_table(distribution)
        max_len = int(self.batch_size * self.walk_len * (
            (1 + self.win_size) - 0.3))
        for walks in self.walk_generator():
            try:
                src_list, pos_list = [], []
                for walk in walks:
                    s, p = skip_gram_gen_pair(walk, self.win_size)
                    src_list.append(s[:max_len]), pos_list.append(p[:max_len])
                src = [s for x in src_list for s in x]
                pos = [s for x in pos_list for s in x]
                src = np.array(src, dtype=np.int64),
                pos = np.array(pos, dtype=np.int64)
                src, pos = np.reshape(src, [-1, 1, 1]), np.reshape(pos,
                                                                   [-1, 1, 1])

                neg_sample_size = [len(pos), self.neg_num, 1]
                if src.shape[0] == 0:
                    continue
                if self.neg_sample_type == "average":
                    negs = np.random.randint(
                        low=0, high=self.graph.num_nodes, size=neg_sample_size)
                elif self.neg_sample_type == "outdegree":
                    negs = alias_sample(neg_sample_size, alias, events)
                elif self.neg_sample_type == "inbatch":
                    pass
                dst = np.concatenate([pos, negs], 1)
                # [batch_size, 1, 1] [batch_size, neg_num+1, 1]
                yield src[:max_len], dst[:max_len]
            except Exception as e:
                log.exception(e)
