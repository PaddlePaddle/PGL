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

from pgl.layers.conv import gcn 
from pgl import graph
from pgl import graph_wrapper


class BatchedGraphWrapper(unittest.TestCase):
    """BatchedGraphWrapper
    """
    def test_batched_graph_wrapper(self):
        """test_batch_graph_wrapper
        """
        np.random.seed(1)

        graph_list = []
      
        num_graph = 5 
        feed_num_nodes = []
        feed_num_edges = []
        feed_edges = []
        feed_node_feats = []
        
        for _ in range(num_graph):
            num_nodes = np.random.randint(5, 20) 
            edges = np.random.randint(low=0, high=num_nodes, size=(10, 2))
            node_feat = {"feature": np.random.rand(num_nodes, 4).astype("float32")}
            single_graph  = graph.Graph(num_nodes=num_nodes, edges=edges, node_feat=node_feat)
            feed_num_nodes.append(num_nodes)
            feed_num_edges.append(len(edges))
            feed_edges.append(edges)
            feed_node_feats.append(node_feat["feature"])
            graph_list.append(single_graph)

        multi_graph = graph.MultiGraph(graph_list)
 
        np.random.seed(1)
        hidden_size = 8
        num_nodes = 10

        place = F.CUDAPlace(0)# if use_cuda else F.CPUPlace()
        prog = F.Program()
        startup_prog = F.Program()
        
        with F.program_guard(prog, startup_prog):
            with F.unique_name.guard():
                # Standard Graph Wrapper
                gw = graph_wrapper.GraphWrapper(
                    name='graph',
                    place=place,
                    node_feat=[("feature", [-1, 4], "float32")])

                output = gcn(gw,
                    gw.node_feat['feature'],
                    hidden_size=hidden_size,
                    activation='relu',
                    name='gcn')

                # BatchGraphWrapper
                num_nodes = L.data(name="num_nodes", shape=[-1], dtype="int32")
                num_edges= L.data(name="num_edges", shape=[-1], dtype="int32")
                edges = L.data(name="edges", shape=[-1, 2], dtype="int32")
                node_feat = L.data(name="node_feats", shape=[-1, 4], dtype="float32")
                batch_gw = graph_wrapper.BatchGraphWrapper(num_nodes=num_nodes,
                                                 num_edges=num_edges,
                                                 edges=edges,
                                                 node_feats={"feature": node_feat})

                output2 = gcn(batch_gw,
                    batch_gw.node_feat['feature'],
                    hidden_size=hidden_size,
                    activation='relu',
                    name='gcn')
    

        exe = F.Executor(place)
        exe.run(startup_prog)
        feed_dict = gw.to_feed(multi_graph)
        feed_dict["num_nodes"] = np.array(feed_num_nodes, dtype="int32")
        feed_dict["num_edges"] = np.array(feed_num_edges, dtype="int32")
        feed_dict["edges"] = np.array(np.concatenate(feed_edges, 0), dtype="int32").reshape([-1, 2])
        feed_dict["node_feats"] = np.array(np.concatenate(feed_node_feats, 0), dtype="float32").reshape([-1, 4])
        
        # Run
        O1, O2 = exe.run(prog, feed=feed_dict, fetch_list=[output, output2])

        # The output from two kind of models should be same.
        for o1, o2 in zip(O1, O2):
            dist = np.sum((o1 - o2) ** 2)
            self.assertLess(dist, 1e-15)


if __name__ == "__main__":
    unittest.main()
