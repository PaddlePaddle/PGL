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
This file provides the Dataset for LINE model.
"""
import os
import io
import sys
import numpy as np

from pgl import graph
from pgl.utils.logger import log


class FlickrDataset(object):
    """Flickr dataset implementation

    Args:
        name: The name of the dataset.

        symmetry_edges: Whether to create symmetry edges.

        self_loop:  Whether to contain self loop edges.

        train_percentage: The percentage of nodes to be trained in multi class task.

    Attributes:
        graph: The :code:`Graph` data object.

        num_groups: Number of classes.

        train_index: The index for nodes in training set.

        test_index: The index for nodes in validation set.
    """

    def __init__(self,
                 data_path,
                 symmetry_edges=False,
                 self_loop=False,
                 train_percentage=0.5):
        self.path = data_path
        #  self.name = name
        self.num_groups = 5
        self.symmetry_edges = symmetry_edges
        self.self_loop = self_loop
        self.train_percentage = train_percentage
        self._load_data()

    def _load_data(self):
        edge_path = os.path.join(self.path, 'edges.txt')
        node_path = os.path.join(self.path, 'nodes.txt')
        nodes_label_path = os.path.join(self.path, 'nodes_label.txt')

        all_edges = []
        edges_weight = []

        with io.open(node_path) as inf:
            num_nodes = len(inf.readlines())

        node_feature = np.zeros((num_nodes, self.num_groups))

        with io.open(nodes_label_path) as inf:
            for line in inf:
                # group_id means the label of the node
                node_id, group_id = line.strip('\n').split(',')
                node_id = int(node_id) - 1
                labels = group_id.split(' ')
                for i in labels:
                    node_feature[node_id][int(i) - 1] = 1

        node_degree_list = [1 for _ in range(num_nodes)]

        with io.open(edge_path) as inf:
            for line in inf:
                items = line.strip().split('\t')
                if len(items) == 2:
                    u, v = int(items[0]), int(items[1])
                    weight = 1  # binary weight, default set to 1
                else:
                    u, v, weight = int(items[0]), int(items[1]), float(items[
                        2]),
                u, v = u - 1, v - 1
                all_edges.append((u, v))
                edges_weight.append(weight)

                if self.symmetry_edges:
                    all_edges.append((v, u))
                    edges_weight.append(weight)

                # sum the weights of the same node as the outdegree
                node_degree_list[u] += weight

        if self.self_loop:
            for i in range(num_nodes):
                all_edges.append((i, i))
                edges_weight.append(1.)

        all_edges = list(set(all_edges))
        self.graph = graph.Graph(
            num_nodes=num_nodes,
            edges=all_edges,
            node_feat={"group_id": node_feature})

        perm = np.arange(0, num_nodes)
        np.random.shuffle(perm)
        train_num = int(num_nodes * self.train_percentage)
        self.train_index = perm[:train_num]
        self.test_index = perm[train_num:]

        edge_distribution = np.array(edges_weight, dtype=np.float32)
        self.edge_distribution = edge_distribution / np.sum(edge_distribution)
        self.edge_sampling = AliasSampling(prob=edge_distribution)

        node_dist = np.array(node_degree_list, dtype=np.float32)
        node_negative_distribution = np.power(node_dist, 0.75)
        self.node_negative_distribution = node_negative_distribution / np.sum(
            node_negative_distribution)
        self.node_sampling = AliasSampling(prob=node_negative_distribution)

        self.node_index = {}
        self.node_index_reversed = {}
        for index, e in enumerate(self.graph.edges):
            self.node_index[e[0]] = index
            self.node_index_reversed[index] = e[0]

    def fetch_batch(self,
                    batch_size=16,
                    K=10,
                    edge_sampling='alias',
                    node_sampling='alias'):
        """Fetch batch data from dataset.
        """
        if edge_sampling == 'numpy':
            edge_batch_index = np.random.choice(
                self.graph.num_edges,
                size=batch_size,
                p=self.edge_distribution)
        elif edge_sampling == 'alias':
            edge_batch_index = self.edge_sampling.sampling(batch_size)
        elif edge_sampling == 'uniform':
            edge_batch_index = np.random.randint(
                0, self.graph.num_edges, size=batch_size)
        u_i = []
        u_j = []
        label = []
        for edge_index in edge_batch_index:
            edge = self.graph.edges[edge_index]
            u_i.append(edge[0])
            u_j.append(edge[1])
            label.append(1)
            for i in range(K):
                while True:
                    if node_sampling == 'numpy':
                        negative_node = np.random.choice(
                            self.graph.num_nodes,
                            p=self.node_negative_distribution)
                    elif node_sampling == 'alias':
                        negative_node = self.node_sampling.sampling()
                    elif node_sampling == 'uniform':
                        negative_node = np.random.randint(0,
                                                          self.graph.num_nodes)

                    # make sure the sampled node has no edge with the source node
                    if not self.graph.has_edges_between(
                            np.array(
                                [self.node_index_reversed[negative_node]]),
                            np.array([self.node_index_reversed[edge[0]]])):
                        break
                u_i.append(edge[0])
                u_j.append(negative_node)
                label.append(-1)
        u_i = np.array([u_i], dtype=np.int64).T
        u_j = np.array([u_j], dtype=np.int64).T
        label = np.array(label, dtype=np.float32)
        return u_i, u_j, label


class AliasSampling:
    """Implemention of Alias-Method

    This is an implementation of Alias-Method for sampling efficiently from 
    a discrete probability distribution.

    Reference: https://en.wikipedia.org/wiki/Alias_method

    Args:
        prob: The discrete probability distribution.

    """

    def __init__(self, prob):
        self.n = len(prob)
        self.U = np.array(prob) * self.n
        self.K = [i for i in range(len(prob))]
        overfull, underfull = [], []
        for i, U_i in enumerate(self.U):
            if U_i > 1:
                overfull.append(i)
            elif U_i < 1:
                underfull.append(i)
        while len(overfull) and len(underfull):
            i, j = overfull.pop(), underfull.pop()
            self.K[j] = i
            self.U[i] = self.U[i] - (1 - self.U[j])
            if self.U[i] > 1:
                overfull.append(i)
            elif self.U[i] < 1:
                underfull.append(i)

    def sampling(self, n=1):
        """Sampling.
        """
        x = np.random.rand(n)
        i = np.floor(self.n * x)
        y = self.n * x - i
        i = i.astype(np.int64)
        res = [i[k] if y[k] < self.U[i[k]] else self.K[i[k]] for k in range(n)]
        if n == 1:
            return res[0]
        else:
            return res
