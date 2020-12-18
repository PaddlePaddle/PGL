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
    This file is for testing gin layer.
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import unittest
import numpy as np

import paddle.fluid as F
import paddle.fluid.layers as L

from pgl.layers.conv import gin
from pgl import graph
from pgl import graph_wrapper


class GinTest(unittest.TestCase):
    """GinTest
    """

    def test_gin(self):
        """test_gin
        """
        np.random.seed(1)
        hidden_size = 8

        num_nodes = 10

        edges = [(1, 4), (0, 5), (1, 9), (1, 8), (2, 8), (2, 5), (3, 6),
                 (3, 7), (3, 4), (3, 8)]
        inver_edges = [(v, u) for u, v in edges]
        edges.extend(inver_edges)

        node_feat = {"feature": np.random.rand(10, 4).astype("float32")}

        g = graph.Graph(num_nodes=num_nodes, edges=edges, node_feat=node_feat)

        use_cuda = False
        place = F.CUDAPlace(0) if use_cuda else F.CPUPlace()

        prog = F.Program()
        startup_prog = F.Program()
        with F.program_guard(prog, startup_prog):
            gw = graph_wrapper.GraphWrapper(
                name='graph',
                place=place,
                node_feat=g.node_feat_info(),
                edge_feat=g.edge_feat_info())

            output = gin(gw,
                         gw.node_feat['feature'],
                         hidden_size=hidden_size,
                         activation='relu',
                         name='gin',
                         init_eps=1,
                         train_eps=True)

        exe = F.Executor(place)
        exe.run(startup_prog)
        ret = exe.run(prog, feed=gw.to_feed(g), fetch_list=[output])

        self.assertEqual(ret[0].shape[0], num_nodes)
        self.assertEqual(ret[0].shape[1], hidden_size)


if __name__ == "__main__":
    unittest.main()
