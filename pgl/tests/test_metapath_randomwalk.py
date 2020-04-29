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
"""test_metapath_randomwalk"""
import time
import unittest
import json
import os

import numpy as np
from pgl.sample import metapath_randomwalk
from pgl.graph import Graph
from pgl import heter_graph

np.random.seed(1)


class MetapathRandomwalkTest(unittest.TestCase):
    """metapath_randomwalk test
    """

    def setUp(self):
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

        self.node_types = ['c' for _ in range(4)] + [
            'p' for _ in range(6)
        ] + ['a' for _ in range(5)]
        node_types = [(i, t) for i, t in enumerate(self.node_types)]

        self.graph = heter_graph.HeterGraph(
            num_nodes=len(node_types), edges=edges, node_types=node_types)

    def test_metapath_randomwalk(self):
        meta_path = 'c2p-p2a-a2p-p2c'
        path = ['c', 'p', 'a', 'p', 'c']
        start_nodes = [0, 1, 2, 3]
        walk_len = 10
        walks = metapath_randomwalk(
            graph=self.graph,
            start_nodes=start_nodes,
            metapath=meta_path,
            walk_length=walk_len)

        self.assertEqual(len(walks), 4)

        for walk in walks:
            for i in range(len(walk)):
                idx = i % (len(path) - 1)
                self.assertEqual(self.node_types[walk[i]], path[idx])


if __name__ == "__main__":
    unittest.main()
