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

    def __init__(self, graph=None, name=None):
        super(GraphTensor, self).__init__()
        self._name = name
        self._graph_attr_holder = [
            "_edges_dst",
            "_edges_src",
            "_edge_uniq_dst",
            "_edge_uniq_dst_count",
            "_graph_lod",
            "_indegree",
            "_num_edges",
            "_num_graph",
            "_num_nodes",
        ]

        if graph is not None:
            self.to_tensor(graph)
        else:
            for attr in self._graph_attr_holder:
                setattr(self, attr, None)

    def __set_attrs(self, tensors):
        assert len(self._graph_attr_holder) == len(
            tensors), "Value errors, the tensors mismatch the graph."
        for attr, value in zip(self._graph_attr_holder, tensors):
            setattr(self, attr, value)
            prefix = "_edge_feat_"
            if attr.startswith(prefix):
                self.edge_feat_tensor_dict[attr[len(prefix):]] = value
            prefix = "_node_feat_"
            if attr.startswith("_node_feat_"):
                self.node_feat_tensor_dict[attr[len(prefix):]] = value

    def to_tensor(self, graph):
        def map_func(var):
            return paddle.to_tensor(var)

        attrs = self.__create_graph_attr(graph, map_func)
        self._graph_attr_holder = list(attrs.keys())
        self.__set_attrs(attrs.values())
        return list(attrs.values())

    def to_numpy(self, graph):
        def map_func(var):
            return np.array(var)

        attrs = self.__create_graph_attr(graph, map_func)
        self._graph_attr_holder = list(attrs.keys())
        return list(attrs.values())

    def from_tensor(self, tensors):
        self.__set_attrs(tensors)
        return self

    def __create_graph_attr(self, graph, map_func=None):
        """Create graph attributes for paddlepaddle.
        """
        assert isinstance(
            graph, pgl.graph.
            Graph), "The input graph should be pgl.graph.Graph instance."
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

        attrs = dict()
        attrs["_edges_src"] = map_func(src)
        attrs["_edges_dst"] = map_func(dst)
        attrs["_edge_uniq_dst"] = map_func(uniq_dst)
        attrs["_edge_uniq_dst_count"] = map_func(uniq_dst_count)
        attrs["_indegree"] = map_func(indegree)
        attrs["_graph_lod"] = map_func(graph_lod)
        attrs["_num_edges"] = map_func(np.array([num_edges], dtype="int64"))
        attrs["_num_graph"] = map_func(np.array([num_graph], dtype="int64"))
        attrs["_num_nodes"] = map_func(np.array([graph.num_nodes]))

        for node_feat_name, node_feat_value in node_feat.items():
            setattr(self, "_node_feat_" + node_feat_name, None)
            attrs["_node_feat_" + node_feat_name] = map_func(node_feat_value)

        for edge_feat_name, edge_feat_value in edge_feat.items():
            setattr(self, "_edge_feat_" + edge_feat_name, None)
            attrs[self, "_edge_feat_" + edge_feat_name] = map_func(
                edge_feat_value)

        return attrs
