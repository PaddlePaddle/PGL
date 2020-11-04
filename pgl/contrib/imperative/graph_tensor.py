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
import paddle
import paddle.fluid as fluid

import numpy as np

import pgl
from pgl.utils import op
from pgl.utils import paddle_helper
from pgl.math import segment_sum, segment_mean, segment_sum, segment_max

import warnings


class MessagePassing(object):
    def __init__(self, graph_tensor, messages):
        self._segment_ids = graph_tensor.dst
        self._messages = messages

    def reduce_sum(self, msg):
        return segment_sum(msg, self._segment_ids)

    def reduce_mean(self, msg):
        return segment_mean(msg, self._segment_ids)

    def reduce_max(self, msg):
        return segment_max(msg, self._segment_ids)

    def reduce_min(self, msg):
        return segment_max(msg, self._segment_ids)

    def sequence_expand(self, msg):
        return paddle.gather(msg, dst, axis=0)

    def reduce_sofmax(self, msg, beta=None):
        if beta is not None:
            msg = msg * beta
        msg_max = self.reduce_max(msg)
        msg_max = self.sequence_expand(msg_max)
        msg = msg - msg_max
        exp_msg = paddle.fluid.layers.exp(msg)

        sum_exp_x = self.reduce_sum(exp_msg)
        sum_exp_x = self.sequence_expand(sum_exp_x)

        return exp_x / sum_exp_x

    def __getitem__(self, key):
        return self._messages[key]


def send(src, dst, nfeat, efeat, message_func):
    """Send message from src to dst.
    """
    src_feat = op.RowReader(nfeat, src)
    dst_feat = op.RowReader(nfeat, dst)
    msg = message_func(src_feat, dst_feat, efeat)
    return msg


def recv(dst, uniq_dst, msg, reduce_function, num_nodes, num_edges):
    """Recv message from given msg to dst nodes.
    """
    if type(reduce_function) is str:
        if isinstance(msg, dict):
            raise TypeError("The message for build-in function"
                            " should be Tensor not dict.")

        if reduce_function == "sum":
            # out_dim = msg.shape[-1]
            # init_output = fluid.layers.fill_constant(
            #     shape=[num_nodes, out_dim], value=0, dtype=msg.dtype)
            # init_output.stop_gradient = True
            # # empty_msg_flag = fluid.layers.cast(num_edges > 0, dtype=msg.dtype)
            # # msg = msg * empty_msg_flag
            # output = paddle_helper.scatter_add(init_output, dst, msg)
            output = segment_sum(msg, dst)
        elif reduce_function == "mean":
            output = segment_mean(msg, dst)
        elif reduce_function == "max":
            output = segment_max(msg, dst)
        elif reduce_function == "min":
            output = segment_min(msg, dst)
        else:
            raise TypeError("Unsuport reduce function!")
    else:
        # output = reduce_function(dst, msg)
        output = reduce_function(msg)

    # TODO(@ZHUI) fix when last node do not have coresponding source node.
    # final_output = fluid.layers.scatter(init_output, uniq_dst, output)
    return output


class GraphTensor(pgl.graph_wrapper.BaseGraphWrapper):
    """Copy the graph object to anonymous shared memory.
    """

    def __init__(self, graph):
        super(GraphTensor, self).__init__()
        self.__create_graph_attr(graph)

    def recv(self, msg, reduce_function):
        output = recv(
            dst=self._edges_dst,
            uniq_dst=self._edge_uniq_dst,
            msg=msg,
            reduce_function=reduce_function,
            num_edges=self._num_edges,
            num_nodes=self._num_nodes)
        return output

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
