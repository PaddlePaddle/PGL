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

import pgl
from pgl.utils import op
from pgl.utils import paddle_helper
from pgl.math import segment_sum, segment_mean, segment_sum, segment_max

import warnings


def send(src, dst, nfeat, efeat, message_func):
    """Send message from src to dst.
    """
    src_feat = op.read_rows(nfeat, src)
    dst_feat = op.read_rows(nfeat, dst)
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


class GraphTensor(pgl.graph.Graph):
    """Copy the graph object to anonymous shared memory.
    """

    def __init__(self, graph):
        degree = graph.indegree()

        #degree = graph.outdegree()
        def share_feat(feat):
            for key in feat:
                feat[key] = paddle.to_tensor(feat[key])
            return feat

        def share_adj_index(index):
            if index is not None:
                index._degree = paddle.to_tensor(index._degree)
                index._sorted_u = paddle.to_tensor(index._sorted_u)
                index._sorted_v = paddle.to_tensor(index._sorted_v)
                index._sorted_eid = paddle.to_tensor(index._sorted_eid)
                index._indptr = paddle.to_tensor(index._indptr)
            return index

        self._num_nodes = graph._num_nodes
        # src, dst, eid = graph.sorted_edges(sort_by="dst")
        # self._edges_src = paddle.to_tensor(src)
        # self._edges_dst = paddle.to_tensor(dst)
        self._uniq_dst = paddle.to_tensor(graph.nodes[graph.indegree() > 0])
        self._edges_src = paddle.to_tensor(graph._edges[:, 0])
        self._edges_dst = paddle.to_tensor(graph._edges[:, 1])
        # self._edges = paddle.to_tensor(graph._edges)
        self._adj_src_index = share_adj_index(graph._adj_src_index)
        self._adj_dst_index = share_adj_index(graph._adj_dst_index)
        self._node_feat = share_feat(graph._node_feat)
        self._edge_feat = share_feat(graph._edge_feat)
        self._num_edges = len(graph._edges)

        # self.src, self.dst = self.edges
        # self.dst, self.src = self._adj_src_index._sorted_u, self._adj_src_index._sorted_v
        # self.dst, self.src = self._adj_dst_index._sorted_u, self._adj_dst_index._sorted_v
        self.dst, self.src = self._adj_dst_index._sorted_u, self._adj_dst_index._sorted_v

    def send(self, message_func, nfeat_list=None, efeat_list=None):
        if efeat_list is None:
            efeat_list = {}
        if nfeat_list is None:
            nfeat_list = {}

        nfeat = {}
        for feat in nfeat_list:
            if isinstance(feat, str):
                nfeat[feat] = self.node_feat[feat]
            else:
                name, tensor = feat
                nfeat[name] = tensor

        efeat = {}
        for feat in efeat_list:
            if isinstance(feat, str):
                efeat[feat] = self.edge_feat[feat]
            else:
                name, tensor = feat
                efeat[name] = tensor

        msg = send(self.src, self.dst, nfeat, efeat, message_func)
        return msg

    def recv(self, msg, reduce_function):
        output = recv(
            # dst=self._edges_dst,
            dst=self.dst,
            uniq_dst=self._uniq_dst,
            msg=msg,
            reduce_function=reduce_function,
            num_edges=self._num_edges,
            num_nodes=self._num_nodes)
        return output

    @property
    def edges(self):
        return self._edges_src, self._edges_dst
