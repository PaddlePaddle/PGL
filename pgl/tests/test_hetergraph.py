# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""test_hetergraph"""

import time
import unittest
import json
import os

import numpy as np
from pgl.sample import metapath_randomwalk
from pgl.graph import Graph
from pgl import heter_graph


class HeterGraphTest(unittest.TestCase):
    """HeterGraph test
    """

    @classmethod
    def setUpClass(cls):
        np.random.seed(1)
        edges = {}
        # for test no successor
        edges['c2p'] = [(1, 4), (0, 5), (1, 9), (1, 8), (2, 8), (2, 5), (3, 6),
                        (3, 7), (3, 4), (3, 8)]
        edges['p2c'] = [(v, u) for u, v in edges['c2p']]
        edges['p2a'] = [(4, 10), (4, 11), (4, 12), (4, 14), (4, 13), (6, 12),
                        (6, 11), (6, 14), (7, 12), (7, 11), (8, 14), (9, 10)]
        edges['a2p'] = [(v, u) for u, v in edges['p2a']]

        # for test speed
        #  edges['c2p'] = [(0, 4), (0, 5), (1, 9), (1,8), (2,8), (2,5), (3,6), (3,7), (3,4), (3,8)]
        #  edges['p2c'] = [(v,u) for u, v in edges['c2p']]
        #  edges['p2a'] = [(4,10), (4,11), (4,12), (4,14), (5,13), (6,13), (6,11), (6,14), (7,12), (7,11), (8,14), (9,13)]
        #  edges['a2p'] = [(v,u) for u, v in edges['p2a']]

        node_types = ['c' for _ in range(4)] + ['p' for _ in range(6)
                                                ] + ['a' for _ in range(5)]
        node_types = [(i, t) for i, t in enumerate(node_types)]

        cls.graph = heter_graph.HeterGraph(
            num_nodes=len(node_types), edges=edges, node_types=node_types)

    def test_num_nodes_by_type(self):
        print()
        n_types = {'c': 4, 'p': 6, 'a': 5}
        for nt in n_types:
            num_nodes = self.graph.num_nodes_by_type(nt)
            self.assertEqual(num_nodes, n_types[nt])

    def test_node_batch_iter(self):
        print()
        batch_size = 2
        ground = [[4, 5], [6, 7], [8, 9]]
        for idx, nodes in enumerate(
                self.graph.node_batch_iter(
                    batch_size=batch_size, shuffle=False, n_type='p')):
            self.assertEqual(len(nodes), batch_size)
            self.assertListEqual(list(nodes), ground[idx])

    def test_sample_successor(self):
        print()
        nodes = [4, 5, 8]
        md = 2
        succes = self.graph.sample_successor(
            edge_type='p2a', nodes=nodes, max_degree=md, return_eids=False)
        self.assertIsInstance(succes, list)
        ground = [[10, 11, 12, 14, 13], [], [14]]
        for succ, g in zip(succes, ground):
            self.assertIsInstance(succ, np.ndarray)
            for i in succ:
                self.assertIn(i, g)

        nodes = [4]
        succes = self.graph.sample_successor(
            edge_type='p2a', nodes=nodes, max_degree=md, return_eids=False)
        self.assertIsInstance(succes, list)
        ground = [[10, 11, 12, 14, 13]]
        for succ, g in zip(succes, ground):
            self.assertIsInstance(succ, np.ndarray)
            for i in succ:
                self.assertIn(i, g)

    def test_successor(self):
        print()
        nodes = [4, 5, 8]
        e_type = 'p2a'
        succes = self.graph.successor(
            edge_type=e_type,
            nodes=nodes, )

        self.assertIsInstance(succes, np.ndarray)
        ground = [[10, 11, 12, 14, 13], [], [14]]
        for succ, g in zip(succes, ground):
            self.assertIsInstance(succ, np.ndarray)
            self.assertCountEqual(succ, g)

        nodes = [4]
        e_type = 'p2a'
        succes = self.graph.successor(
            edge_type=e_type,
            nodes=nodes, )

        self.assertIsInstance(succes, np.ndarray)
        ground = [[10, 11, 12, 14, 13]]
        for succ, g in zip(succes, ground):
            self.assertIsInstance(succ, np.ndarray)
            self.assertCountEqual(succ, g)

    def test_sample_nodes(self):
        print()
        p_ground = [4, 5, 6, 7, 8, 9]
        sample_num = 10
        nodes = self.graph.sample_nodes(sample_num=sample_num, n_type='p')

        self.assertEqual(len(nodes), sample_num)
        for n in nodes:
            self.assertIn(n, p_ground)

        # test n_type == None
        ground = [i for i in range(15)]
        nodes = self.graph.sample_nodes(sample_num=sample_num, n_type=None)
        self.assertEqual(len(nodes), sample_num)
        for n in nodes:
            self.assertIn(n, ground)


if __name__ == "__main__":
    unittest.main()
