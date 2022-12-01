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


class GCNConv(BaseConv):
    def __init__(self, input_size, output_size, activation=None, norm=True):
        super(GCNConv, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.linear = nn.Linear(input_size, output_size, bias_attr=False)
        self.bias = self.create_parameter(shape=[output_size], is_bias=True)
        self.norm = norm
        if isinstance(activation, str):
            activation = getattr(F, activation)
        self.activation = activation

    def forward(self, edge_index, num_nodes, feature, norm=None):

        if self.norm and norm is None:
            norm = gcn_norm(edge_index, num_nodes)

        if self.input_size > self.output_size:
            feature = self.linear(feature)

        if norm is not None:
            feature = feature * norm

        output = self.send_recv(edge_index, feature, "sum")

        if self.input_size <= self.output_size:
            output = self.linear(output)

        if norm is not None:
            output = output * norm
        output = output + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output


class GCN(nn.Layer):
    """Implementation of GCN Model
    """

    def __init__(self,
                 input_size,
                 num_class,
                 num_layers=1,
                 hidden_size=64,
                 dropout=0.5):
        super(GCN, self).__init__()
        self.num_class = num_class
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.gcns = nn.LayerList()
        for i in range(self.num_layers):
            if i == 0:
                self.gcns.append(
                    GCNConv(
                        input_size,
                        self.hidden_size,
                        activation="relu",
                        norm=True))
            else:
                self.gcns.append(
                    GCNConv(
                        self.hidden_size,
                        self.hidden_size,
                        activation="relu",
                        norm=True))
            self.gcns.append(nn.Dropout(self.dropout))
        self.gcns.append(GCNConv(self.hidden_size, self.num_class))

    def forward(self, edge_index, num_nodes, feature):
        for m in self.gcns:
            if isinstance(m, nn.Dropout):
                feature = m(feature)
            else:
                feature = m(edge_index, num_nodes, feature)
        return feature
