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


class SGCConv(BaseConv):
    def __init__(self, input_size, output_size, k_hop=2, cached=True, activation=None, bias=False):
        super(SGCConv, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.k_hop = k_hop
        self.linear = nn.Linear(input_size, output_size, bias_attr=False)
        if bias:
            self.bias = self.create_parameter(shape=[output_size], is_bias=True)

        self.cached = cached
        self.cached_output = None
        if isinstance(activation, str):
            activation = getattr(F, activation)
        self.activation = activation

    def forward(self, edge_index, num_nodes, feature):
        if self.cached:
            if self.cached_output is None:
                norm = gcn_norm(edge_index, num_nodes)
                for hop in range(self.k_hop):
                    feature = feature * norm
                    feature = self.send_recv(edge_index, feature, "sum")
                    feature = feature * norm
                self.cached_output = feature
            else:
                feature = self.cached_output
        else:
            norm = gcn_norm(edge_index, num_nodes)
            for hop in range(self.k_hop):
                feature = feature * norm
                feature = self.send_recv(edge_index, feature, "sum")
                feature = feature * norm

        output = self.linear(feature)
        if hasattr(self, "bias"):
            output = output + self.bias

        if self.activation is not None:
            output = self.activation(output)
        return output


class SGC(nn.Layer):
    def __init__(self, input_size, num_class, num_layers=1, **kwargs):
        super(SGC, self).__init__()
        self.num_class = num_class
        self.num_layers = num_layers
        self.sgc_layer = SGCConv(
            input_size=input_size, output_size=num_class, k_hop=num_layers)

    def forward(self, edge_index, num_nodes, feature):
        feature = self.sgc_layer(edge_index, num_nodes, feature)
        return feature
