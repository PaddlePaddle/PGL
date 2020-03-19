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
import sys
import os
import numpy as np
import pickle as pkl
import tqdm
import time
import random
from pgl.utils.logger import log
from pgl import heter_graph


class m2vGraph(object):
    """Implemetation of graph in order to sample metapath random walk.
    """

    def __init__(self, config):
        self.edge_path = config.edge_path
        self.num_nodes = config.num_nodes
        self.symmetry = config.symmetry
        edge_files = config.edge_files
        node_types_file = config.node_types_file

        self.edge_file_list = []
        for pair in edge_files.split(','):
            e_type, filename = pair.split(':')
            filename = os.path.join(self.edge_path, filename)
            self.edge_file_list.append((e_type, filename))

        self.node_types_file = os.path.join(self.edge_path, node_types_file)

        self.build_graph()

    def build_graph(self):
        """Build pgl heterogeneous graph.
        """
        edges_by_types = {}
        npy = self.edge_file_list[0][1] + ".npy"
        if os.path.exists(npy):
            log.info("load data from numpy file")

            for pair in self.edge_file_list:
                edges_by_types[pair[0]] = np.load(pair[1] + ".npy")

        else:
            log.info("load data from txt file")
            for pair in self.edge_file_list:
                edges_by_types[pair[0]] = self.load_edges(pair[1])
                #  np.save(pair[1] + ".npy", edges_by_types[pair[0]])

        for e_type, edges in edges_by_types.items():
            log.info(["number of %s edges: " % e_type, len(edges)])

        if self.symmetry:
            tmp = {}
            for key, edges in edges_by_types.items():
                n_list = key.split('2')
                re_key = n_list[1] + '2' + n_list[0]
                tmp[re_key] = edges_by_types[key][:, [1, 0]]
            edges_by_types.update(tmp)

        log.info(["finished loadding symmetry edges."])

        node_types = self.load_node_types(self.node_types_file)

        assert len(node_types) == self.num_nodes, \
                "num_nodes should be equal to the length of node_types"
        log.info(["number of nodes: ", len(node_types)])

        node_features = {
            'index': np.array([i for i in range(self.num_nodes)]).reshape(
                -1, 1).astype(np.int64)
        }

        self.graph = heter_graph.HeterGraph(
            num_nodes=self.num_nodes,
            edges=edges_by_types,
            node_types=node_types,
            node_feat=node_features)

    def load_edges(self, file_, symmetry=False):
        """Load edges from file.
        """
        edges = []
        with open(file_, 'r') as reader:
            for line in reader:
                items = line.strip().split()
                src, dst = int(items[0]), int(items[1])
                edges.append((src, dst))
                if symmetry:
                    edges.append((dst, src))
            edges = np.array(list(set(edges)), dtype=np.int64)
            #  edges = list(set(edges))
        return edges

    def load_node_types(self, file_):
        """Load node types 
        """
        node_types = []
        log.info("node_types_file name: %s" % file_)
        with open(file_, 'r') as reader:
            for line in reader:
                items = line.strip().split()
                node_id = int(items[0])
                n_type = items[1]
                node_types.append((node_id, n_type))

        return node_types
