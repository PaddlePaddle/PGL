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
"""test_redis_graph"""
import time
import unittest
import json
import os

import numpy as np
from pgl.redis_graph import RedisGraph


class RedisGraphTest(unittest.TestCase):
    """RedisGraphTest
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

    def test_random_seed(self):
        """test_random_seed
        """
        np.random.seed(1)
        data1 = self.graph.sample_predecessor(range(1000), max_degree=5)
        data1 = [nid for nodes in data1 for nid in nodes]
        np.random.seed(1)
        data2 = self.graph.sample_predecessor(range(1000), max_degree=5)
        data2 = [nid for nodes in data2 for nid in nodes]
        np.random.seed(3)
        data3 = self.graph.sample_predecessor(range(1000), max_degree=5)
        data3 = [nid for nodes in data3 for nid in nodes]

        self.assertEqual(data1, data2)
        self.assertNotEqual(data2, data3)


if __name__ == '__main__':
    unittest.main()
