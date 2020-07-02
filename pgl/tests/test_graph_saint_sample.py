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
"""graph saint sample test
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import unittest
import numpy as np

import pgl
import paddle.fluid as fluid
from pgl.sample import graph_saint_random_walk_sample


class GraphSaintSampleTest(unittest.TestCase):
    """GraphSaintSampleTest"""

    def test_randomwalk_sampler(self):
        """test_randomwalk_sampler"""
        g = pgl.graph.Graph(
            num_nodes=8,
            edges=[(1, 2), (2, 3), (0, 2), (0, 1), (6, 7), (4, 5), (6, 4),
                   (7, 4), (3, 4)])
        subgraph = graph_saint_random_walk_sample(g, [6, 7], 2)
        print('reindex', subgraph._from_reindex)
        print('subedges', subgraph.edges)
        assert len(subgraph.nodes) == 4
        assert len(subgraph.edges) == 4
        true_edges = np.array([[0, 1], [2, 3], [2, 0], [3, 0]])
        assert "{}".format(subgraph.edges) == "{}".format(true_edges)


if __name__ == '__main__':
    unittest.main()
