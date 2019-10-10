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
    This package implement graph sampling algorithm.
"""
import unittest
import os
import json

import numpy as np
from pgl.redis_graph import RedisGraph
from pgl.sample import graphsage_sample
from pgl.sample import node2vec_sample


class SampleTest(unittest.TestCase):
    """SampleTest
    """

    def setUp(self):
        config_path = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            'test_redis_graph_conf.json')
        with open(config_path) as inf:
            config = json.load(inf)
        redis_configs = [config["redis"], ]
        self.graph = RedisGraph(
            "reddit-graph", redis_configs, num_parts=config["num_parts"])

    def test_graphsage_sample(self):
        """test_graphsage_sample
        """
        eids = np.random.choice(self.graph.num_edges, 1000)
        edges = self.graph.get_edges_by_id(eids)
        nodes = [n for edge in edges for n in edge]
        ignore_edges = edges.tolist() + edges[:, [1, 0]].tolist()

        np.random.seed(1)
        subgraphs = graphsage_sample(self.graph, nodes, [10, 10], [])

        np.random.seed(1)
        subgraphs_ignored = graphsage_sample(self.graph, nodes, [10, 10],
                                             ignore_edges)

        self.assertEqual(subgraphs[0].num_nodes,
                         subgraphs_ignored[0].num_nodes)
        self.assertGreaterEqual(subgraphs[0].num_edges,
                                subgraphs_ignored[0].num_edges)

    def test_node2vec_sample(self):
        """test_node2vec_sample
        """
        walks = node2vec_sample(self.graph, range(10), 3)


if __name__ == '__main__':
    unittest.main()
