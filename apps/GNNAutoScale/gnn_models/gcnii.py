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

import numpy as np
import paddle
import pgl
import paddle.nn as nn

from .base_model import ScalableGNN


class GCNII(ScalableGNN):
    def __init__(self,
                 num_nodes,
                 num_layers,
                 input_size,
                 hidden_size,
                 output_size,
                 alpha=0.1,
                 lambda_l=0.5,
                 dropout=0.0,
                 drop_input=True,
                 pool_size=None,
                 buffer_size=None,
                 **kwargs):
        super().__init__(num_nodes, num_layers, hidden_size, pool_size,
                         buffer_size)

        self.input_size = input_size
        self.output_size = output_size
        self.alpha = alpha
        self.lambda_l = lambda_l
        self.dropout_fn = nn.Dropout(p=dropout)
        self.drop_input = drop_input

        self.linears = nn.LayerList()
        self.linears.append(nn.Linear(input_size, hidden_size))
        self.linears.append(nn.Linear(hidden_size, output_size))

        self.mlps = nn.LayerList()
        for _ in range(num_layers):
            self.mlps.append(nn.Linear(hidden_size, hidden_size))

    def mini_gcnii(self, i, graph, x, norm, x0):
        beta_i = np.log(1.0 * self.lambda_l / i + 1)
        if norm is not None:
            x = x * norm
        x = graph.send_recv(x)
        if norm is not None:
            x = x * norm
        x = self.alpha * x0 + (1 - self.alpha) * x

        x_transed = self.mlps[i - 1](x)
        x = beta_i * x_transed + (1 - beta_i) * x
        return x

    def forward(self, graph, x, norm, *args):
        if self.drop_input:
            x = self.dropout_fn(x)
        x = self.linears[0](x)
        x = x0 = paddle.nn.ReLU()(x)
        x = self.dropout_fn(x)

        for i, hist in enumerate(self.histories):
            x = self.mini_gcnii(i + 1, graph, x, norm, x0)
            x = paddle.nn.ReLU()(x)
            x = self.push_and_pull(hist, x, *args)
            x = self.dropout_fn(x)

        x = self.mini_gcnii(self.num_layers, graph, x, norm, x0)
        x = paddle.nn.ReLU()(x)
        x = self.dropout_fn(x)
        x = self.linears[1](x)
        return x

    @paddle.no_grad()
    def forward_layer(self, layer, graph, x, norm, state):
        if layer == 0:
            if self.drop_input:
                x = self.dropout_fn(x)
            x = self.linears[0](x)
            x = x0 = paddle.nn.ReLU()(x)
            state['x0'] = x0

        x = self.dropout_fn(x)
        x = self.mini_gcnii(layer + 1, graph, x, norm, state['x0'])
        x = paddle.nn.ReLU()(x)

        if layer == self.num_layers - 1:
            x = self.dropout_fn(x)
            x = self.linears[1](x)
        return x
