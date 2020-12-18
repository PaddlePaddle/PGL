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

import unittest
import numpy as np
from pgl.graph import Graph


class GraphTest(unittest.TestCase):
    def setUp(self):
        num_nodes = 5
        edges = [(0, 1), (1, 2), (3, 4)]
        feature = np.random.randn(5, 100)
        edge_feature = np.random.randn(3, 100)
        self.graph = Graph(
            num_nodes=num_nodes,
            edges=edges,
            node_feat={"feature": feature},
            edge_feat={"edge_feature": edge_feature})

    def test_subgraph_consistency(self):
        node_index = [0, 2, 3, 4]
        eid = [2]
        subgraph = self.graph.subgraph(node_index, eid)
        for key, value in subgraph.node_feat.items():
            diff = value - self.graph.node_feat[key][node_index]
            diff = np.sqrt(np.sum(diff * diff))
            self.assertLessEqual(diff, 1e-6)

        for key, value in subgraph.edge_feat.items():
            diff = value - self.graph.edge_feat[key][eid]
            diff = np.sqrt(np.sum(diff * diff))
            self.assertLessEqual(diff, 1e-6)


if __name__ == '__main__':
    unittest.main()
