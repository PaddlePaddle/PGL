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

        output = self.send_recv(feature, "sum")

        if self.input_size <= self.output_size:
            output = self.linear(output)

        if norm is not None:
            output = output * norm
        output = output + self.bias
        if self.activation is None:
            output = self.activation(output)
        return output
