# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
    BaseConv class for GNN model.
"""

import pdb
import paddle
import paddle.nn as nn

from pgl.utils import op
from pgl.message import Message
from pgl.utils.helper import unique_segment


class BaseConv(nn.Layer):
    def __init__(self):
        super(BaseConv, self).__init__()

    def send_recv(self, edge_index, feature, pool_type="sum"):
        src, dst = edge_index[:, 0], edge_index[:, 1]
        return paddle.geometric.send_u_recv(
            feature, src, dst, pool_type=pool_type)

    def send(
            self,
            edge_index,
            message_func,
            src_feat=None,
            dst_feat=None,
            edge_feat=None,
            node_feat=None, ):
        if (src_feat is not None or
                dst_feat is not None) and node_feat is not None:
            raise ValueError(
                "Can not use src/dst feat and node feat at the same time")

        src_feat_temp = {}
        dst_feat_temp = {}
        if node_feat is not None:
            assert isinstance(node_feat,
                              dict), "The input node_feat must be a dict"
            src_feat_temp.update(node_feat)
            dst_feat_temp.update(node_feat)
        else:
            if src_feat is not None:
                assert isinstance(src_feat,
                                  dict), "The input src_feat must be a dict"
                src_feat_temp.update(src_feat)
            if dst_feat is not None:
                assert isinstance(dst_feat,
                                  dict), "The input dst_feat must be a dict"
                dst_feat_temp.update(dst_feat)

        edge_feat_temp = {}
        if edge_feat is not None:
            assert isinstance(edge_feat,
                              dict), "The input edge_feat must be a dict"
            edge_feat_temp.update(edge_feat)

        src = edge_index[:, 0]
        dst = edge_index[:, 1]

        src_feat = op.RowReader(src_feat_temp, src)
        dst_feat = op.RowReader(dst_feat_temp, dst)
        msg = message_func(src_feat, dst_feat, edge_feat_temp)

        if not isinstance(msg, dict):
            raise TypeError(
                    "The outputs of the %s function is expected to be a dict, but got %s" \
                            % (message_func.__name__, type(msg)))
        return msg

    def recv(self, edge_index, num_nodes, reduce_func, msg, recv_mode="dst"):
        if not isinstance(msg, dict):
            raise TypeError(
                "The input of msg should be a dict, but receives a %s" %
                (type(msg)))

        if not callable(reduce_func):
            raise TypeError("reduce_func should be callable")

        src, dst, eid = self.sort_edges(
            edge_index, num_nodes, sort_by=recv_mode)

        msg = op.RowReader(msg, eid)

        if recv_mode == "dst":
            # pdb.set_trace()
            # uniq_ind, self._segment_ids = unique_segment(dst)
            uniq_ind, segment_ids = unique_segment(dst)
        elif recv_mode == "src":
            # uniq_ind, segment_ids = unique_segment(src)
            uniq_ind, segment_ids = unique_segment(src)

        bucketed_msg = Message(msg, segment_ids)
        output = reduce_func(bucketed_msg)
        output_dim = output.shape[-1]
        init_output = paddle.zeros(
            shape=[num_nodes, output_dim], dtype=output.dtype)
        final_output = paddle.scatter(init_output, uniq_ind, output)

        return final_output

    def sort_edges(self, edge_index, num_nodes, sort_by="src"):
        """Return sorted edges with different strategies.
        """
        if sort_by not in ["src", "dst"]:
            raise ValueError("sort_by should be in 'src' or 'dst'.")

        if sort_by == "src":
            u = edge_index[:, 0]
            v = edge_index[:, 1]
            self.get_sparse_edge_info(u, v, num_nodes)
            src, dst, eid = self._sorted_u, self._sorted_v, self._sorted_eid
        if sort_by == "dst":
            v = edge_index[:, 0]
            u = edge_index[:, 1]
            self.get_sparse_edge_info(u, v, num_nodes)
            dst, src, eid = self._sorted_u, self._sorted_v, self._sorted_eid
        return src, dst, eid

    def get_sparse_edge_info(self, u, v, num_nodes):
        self._degree = paddle.zeros(shape=[num_nodes], dtype="int64")
        self._degree = paddle.scatter(
            x=self._degree,
            overwrite=False,
            index=u,
            updates=paddle.ones_like(
                u, dtype="int64"))
        self._sorted_eid = paddle.argsort(u)
        self._sorted_u = paddle.gather(u, self._sorted_eid)
        self._sorted_v = paddle.gather(v, self._sorted_eid)
        self._indptr = paddle.concat(
            [
                paddle.zeros(
                    shape=[1, ], dtype=self._degree.dtype),
                paddle.cumsum(self._degree)
            ],
            axis=-1)
