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

import pgl
from pgl import graph
from pgl import graph_wrapper


class GraphGatherTest(unittest.TestCase):
    """GraphGatherTest
    """

    def test_graph_gather(self):
        """test_graph_gather
        """
        np.random.seed(1)

        graph_list = []
      
        num_graph = 10
        for _ in range(num_graph):
            num_nodes = np.random.randint(5, 20) 
            edges = np.random.randint(low=0, high=num_nodes, size=(10, 2))
            node_feat = {"feature": np.random.rand(num_nodes, 4).astype("float32")}
            g = graph.Graph(num_nodes=num_nodes, edges=edges, node_feat=node_feat)
            graph_list.append(g)

        gg = graph.MultiGraph(graph_list)
        

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

            index = L.data(name="index", dtype="int32", shape=[-1])
            feats = pgl.layers.graph_gather(gw, gw.node_feat["feature"], index)


        exe = F.Executor(place)
        exe.run(startup_prog)
        feed_dict = gw.to_feed(gg)
        feed_dict["index"] = np.zeros(num_graph, dtype="int32")
        ret = exe.run(prog, feed=feed_dict, fetch_list=[feats])
        self.assertEqual(list(ret[0].shape), [num_graph, 4])
        for i in range(num_graph):
            dist = (ret[0][i] - graph_list[i].node_feat["feature"][0])
            dist = np.sum(dist ** 2)
            self.assertLess(dist, 1e-15)


if __name__ == "__main__":
    unittest.main()
