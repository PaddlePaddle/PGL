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


class APPNP(ScalableGNN):
    def __init__(self,
                 num_nodes,
                 num_layers,
                 input_size,
                 hidden_size,
                 output_size,
                 alpha=0.1,
                 dropout=0.,
                 pool_size=None,
                 buffer_size=None,
                 **kwargs):
        super().__init__(num_nodes, num_layers, output_size, pool_size,
                         buffer_size)

        self.input_size = input_size
        self.output_size = output_size
        self.alpha = alpha
        self.dropout_fn = nn.Dropout(p=dropout)

        self.linears = nn.LayerList()
        self.linears.append(nn.Linear(input_size, hidden_size))
        self.linears.append(nn.Linear(hidden_size, output_size))

    def forward(self, graph, x, norm, *args):
        x = self.dropout_fn(x)
        x = self.linears[0](x)
        x = paddle.nn.ReLU()(x)
        x = self.dropout_fn(x)
        x = self.linears[1](x)
        x0 = x
        for hist in self.histories:  # appnp
            if norm is not None:
                x = x * norm
            x = graph.send_recv(x)
            if norm is not None:
                x = x * norm
            x = self.alpha * x0 + (1 - self.alpha) * x
            x = self.push_and_pull(hist, x, *args)

        if norm is not None:
            x = x * norm
        x = graph.send_recv(x)
        if norm is not None:
            x = x * norm
        x = self.alpha * x0 + (1 - self.alpha) * x
        return x

    @paddle.no_grad()
    def forward_layer(self, layer, graph, x, norm, state):
        if layer == 0:
            x = self.dropout_fn(x)
            x = self.linears[0](x)
            x = paddle.nn.ReLU()(x)
            x = self.dropout_fn(x)
            x = x0 = self.linears[1](x)
            state['x0'] = x0

        if norm is not None:
            x = x * norm
        x = graph.send_recv(x)
        if norm is not None:
            x = x * norm
        x = self.alpha * state['x0'] + (1 - self.alpha) * x
        return x
