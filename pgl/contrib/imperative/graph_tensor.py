# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
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
import warnings
import numpy as np

import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as L

import pgl
from pgl.graph_wrapper import send, recv
from pgl.utils import op
from pgl.utils import paddle_helper


class GraphTensor(pgl.graph_wrapper.BaseGraphWrapper):
    """Copy the graph object to anonymous shared memory.
    """

    def __init__(self, graph):
        super(GraphTensor, self).__init__()
        self.__create_graph_attr(graph)

    def __create_graph_attr(self, graph):
        """Create graph attributes for paddlepaddle.
        """
        src, dst, eid = graph.sorted_edges(sort_by="dst")
        indegree = graph.indegree()
        nodes = graph.nodes
        uniq_dst = nodes[indegree > 0]
        uniq_dst_count = indegree[indegree > 0]
        uniq_dst_count = np.cumsum(uniq_dst_count, dtype='int32')
        uniq_dst_count = np.insert(uniq_dst_count, 0, 0)
        graph_lod = graph.graph_lod
        num_graph = graph.num_graph

        num_edges = len(src)
        if num_edges == 0:
            # Fake Graph
            src = np.array([0], dtype="int64")
            dst = np.array([0], dtype="int64")
            eid = np.array([0], dtype="int64")
            uniq_dst_count = np.array([0, 1], dtype="int32")
            uniq_dst = np.array([0], dtype="int64")

        edge_feat = {}

        for key, value in graph.edge_feat.items():
            edge_feat[key] = value[eid]
        node_feat = graph.node_feat

        self.__create_graph_node_feat(node_feat)
        self.__create_graph_edge_feat(edge_feat)

        self._num_edges = paddle.to_tensor(
            np.array(
                [num_edges], dtype="int64"))
        self._num_graph = paddle.to_tensor(
            np.array(
                [num_graph], dtype="int64"))
        self._edges_src = paddle.to_tensor(src)
        self._edges_dst = paddle.to_tensor(dst)
        self._num_nodes = paddle.to_tensor(np.array([graph.num_nodes]))
        self._edge_uniq_dst = paddle.to_tensor(uniq_dst)
        self._edge_uniq_dst_count = paddle.to_tensor(uniq_dst_count)
        self._graph_lod = paddle.to_tensor(graph_lod)
        self._indegree = paddle.to_tensor(indegree)

    def __create_graph_node_feat(self, node_feat):
        """Convert node features into paddlepaddle tensor.
        """
        for node_feat_name, node_feat_value in node_feat.items():
            self.node_feat_tensor_dict[node_feat_name] = paddle.to_tensor(
                node_feat_value)

    def __create_graph_edge_feat(self, edge_feat):
        """Convert edge features into paddlepaddle tensor.
        """
        for edge_feat_name, edge_feat_value in edge_feat.items():
            self.edge_feat_tensor_dict[edge_feat_name] = paddle.to_tensor(
                edge_feat_value)
