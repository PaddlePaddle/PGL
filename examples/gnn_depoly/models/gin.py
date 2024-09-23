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
from pgl.math import segment_pool

from .base_conv import BaseConv


class GINConv(BaseConv):
    def __init__(
        self, input_size, output_size, activation=None, init_eps=0.0, train_eps=False
    ):
        super(GINConv, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear1 = nn.Linear(input_size, output_size, bias_attr=True)
        self.linear2 = nn.Linear(output_size, output_size, bias_attr=True)
        self.layer_norm = nn.LayerNorm(output_size)
        if train_eps:
            self.epsilon = self.create_parameter(
                shape=[1, 1],
                dtype="float32",
                default_initializer=nn.initializer.Constant(value=init_eps),
            )
        else:
            self.epsilon = init_eps

        if isinstance(activation, str):
            activation = getattr(F, activation)
        self.activation = activation

    def forward(self, edge_index, feature):
        neigh_feature = self.send_recv(edge_index, feature, "sum")
        output = neigh_feature + feature * (self.epsilon + 1.0)

        output = self.linear1(output)
        output = self.layer_norm(output)

        if self.activation is not None:
            output = self.activation(output)

        output = self.linear2(output)

        return output


class GIN(nn.Layer):
    def __init__(self, input_size, num_class, num_layers, hidden_size, pool_type="sum", dropout_prob=0.0):
        super(GIN, self).__init__()
        self.input_size = input_size
        self.num_class = num_class
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.pool_type = pool_type
        self.dropout_prob = dropout_prob

        self.gin_convs = nn.LayerList()
        self.norms = nn.LayerList()
        self.linears = nn.LayerList()
        self.linears.append(nn.Linear(self.input_size, self.num_class))

        for i in range(self.num_layers):
            if i == 0:
                input_size = self.input_size
            else:
                input_size = self.hidden_size
            gin = GINConv(input_size, self.hidden_size, "relu")
            self.gin_convs.append(gin)
            ln = paddle.nn.LayerNorm(self.hidden_size)
            self.norms.append(ln)
            self.linears.append(nn.Linear(self.hidden_size, self.num_class))
        self.relu = nn.ReLU()


    def forward(self, edge_index, feature):
        feature_list = [feature]
        for i in range(self.num_layers):
            h = self.gin_convs[i](edge_index, feature_list[i])
            h = self.norms[i](h)
            h = self.relu(h)
            feature_list.append(h)

        output = 0
        for i, h in enumerate(feature_list):
            h = F.dropout(h, p=self.dropout_prob)
            output += self.linears[i](h)

        return output