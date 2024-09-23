# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn as nn
import paddle.nn.functional as F

from .base_conv import BaseConv


def gcn_norm(edge_index, num_nodes):
    _, col = edge_index[:, 0], edge_index[:, 1]
    degree = paddle.zeros(shape=[num_nodes], dtype="int64")
    degree = paddle.scatter(
        x=degree,
        index=col,
        updates=paddle.ones_like(
            col, dtype="int64"),
        overwrite=False)
    norm = paddle.cast(degree, dtype=paddle.get_default_dtype())
    norm = paddle.clip(norm, min=1.0)
    norm = paddle.pow(norm, -0.5)
    norm = paddle.reshape(norm, [-1, 1])
    return norm


class APPNPConv(BaseConv):
    def __init__(self, alpha=0.2, k_hop=10, self_loop=False):
        super(APPNPConv, self).__init__()
        self.alpha = alpha
        self.k_hop = k_hop
        self.self_loop = self_loop

    def forward(self, edge_index, num_nodes, feature, norm=None):
        if self.self_loop:
            index = paddle.arange(start=0, end=num_nodes, dtype="int64")
            self_loop_edges = paddle.transpose(paddle.stack((index, index)), [1, 0])

            mask = edge_index[:, 0] != edge_index[:, 1]
            mask_index = paddle.masked_select(paddle.arange(end=edge_index.shape[0]), mask)
            edges = paddle.gather(edge_index, mask_index)  # remove self loop

            edge_index = paddle.concat((self_loop_edges, edges), axis=0)
        if norm is None:
            norm = gcn_norm(edge_index, num_nodes)

        h0 = feature

        for _ in range(self.k_hop):
            feature = feature * norm
            feature = self.send_recv(edge_index, feature, "sum")
            feature = feature * norm
            feature = self.alpha * h0 + (1 - self.alpha) * feature

        return feature


class APPNP(nn.Layer):
    """Implement of APPNP"""

    def __init__(self,
                 input_size,
                 num_class,
                 num_layers=1,
                 hidden_size=64,
                 dropout=0.5,
                 k_hop=10,
                 alpha=0.1,
                 **kwargs):
        super(APPNP, self).__init__()
        self.num_class = num_class
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.alpha = alpha
        self.k_hop = k_hop

        self.mlps = nn.LayerList()
        self.mlps.append(nn.Linear(input_size, self.hidden_size))
        self.drop_fn = nn.Dropout(self.dropout)
        for _ in range(self.num_layers - 1):
            self.mlps.append(nn.Linear(self.hidden_size, self.hidden_size))

        self.output = nn.Linear(self.hidden_size, num_class)
        self.appnp = APPNPConv(alpha=self.alpha, k_hop=self.k_hop)

    def forward(self, edge_index, num_nodes, feature):
        for m in self.mlps:
            feature = self.drop_fn(feature)
            feature = m(feature)
            feature = F.relu(feature)
        feature = self.drop_fn(feature)
        feature = self.output(feature)
        feature = self.appnp(edge_index, num_nodes, feature)
        return feature 
