# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import pgl
import paddle.nn as nn

from .base_model import ScalableGNN


class GCN(ScalableGNN):
    def __init__(self,
                 num_nodes,
                 num_layers,
                 input_size,
                 hidden_size,
                 output_size,
                 dropout=0.0,
                 drop_input=True,
                 pool_size=None,
                 buffer_size=None,
                 **kwargs):
        super().__init__(num_nodes, num_layers, hidden_size, pool_size,
                         buffer_size)

        self.input_size = input_size
        self.output_size = output_size
        self.dropout_fn = nn.Dropout(p=dropout)
        self.drop_input = drop_input

        self.convs = nn.LayerList()
        for i in range(num_layers):
            in_dim = out_dim = hidden_size
            if i == 0:
                in_dim = input_size
            if i == num_layers - 1:
                out_dim = output_size
            conv = pgl.nn.GCNConv(in_dim, out_dim, norm=True)
            self.convs.append(conv)

    def forward(self, graph, x, norm, *args):
        if self.drop_input:
            x = self.dropout_fn(x)

        for conv, hist in zip(self.convs[:-1], self.histories):
            x = conv(graph, x, norm)
            x = paddle.nn.ReLU()(x)
            x = self.push_and_pull(hist, x, *args)
            x = self.dropout_fn(x)

        h = self.convs[-1](graph, x, norm)

        return h

    @paddle.no_grad()
    def forward_layer(self, layer, graph, x, norm, state):
        if layer == 0:
            if self.drop_input:
                x = self.dropout_fn(x)

        h = self.convs[layer](graph, x, norm)

        if layer < self.num_layers - 1:
            h = paddle.nn.ReLU()(h)

        return h
