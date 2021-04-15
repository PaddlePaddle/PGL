# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os

import unittest
import numpy as np
import paddle
import pgl
from pgl.sampling import graphsage_sample
from pgl.sampling import random_walk
from pgl.sampling import node2vec_walk
from pgl.sampling import node2vec_walk_plus

from testsuite import create_random_graph


class SampleTest(unittest.TestCase):
    """SampleTest
    """

    def test_graphsage_sample(self):
        """test_graphsage_sample
        """
        graph = create_random_graph()
        nodes = [1, 2, 3]
        np.random.seed(1)
        subgraphs = graphsage_sample(graph, nodes, [10, 10], [])

    def build_test_graph(self):
        num_nodes = 5
        dim = 4
        edges = [(0, 1), (1, 2), (3, 4), (1, 0), (2, 1), (4, 3)]
        nfeat = np.random.randn(num_nodes, dim)
        efeat = np.random.randn(len(edges), dim)

        g1 = pgl.Graph(
            edges=edges,
            num_nodes=num_nodes,
            node_feat={'nfeat': nfeat},
            edge_feat={'efeat': efeat})
        return g1

    def test_random_walk(self):
        g1 = self.build_test_graph()
        walk_paths = random_walk(g1, [0, 1], 2)

    def test_node2vec_walk(self):
        g1 = self.build_test_graph()
        walk_paths = node2vec_walk(g1, [0, 1], 4, p=0.25, q=0.25)

    def test_node2vec_walk_plus(self):
        g1 = self.build_test_graph()
        walk_paths = node2vec_walk_plus(g1, [0, 1], 4, p=0.25, q=0.25)

if __name__ == '__main__':
    unittest.main()
