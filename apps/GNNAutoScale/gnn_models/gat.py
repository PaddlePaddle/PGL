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


class GAT(ScalableGNN):
    def __init__(self,
                 num_nodes,
                 num_layers,
                 input_size,
                 hidden_size,
                 output_size,
                 num_heads,
                 feat_drop=0.6,
                 attn_drop=0.6,
                 pool_size=None,
                 buffer_size=None,
                 **kwargs):
        super().__init__(num_nodes, num_layers, hidden_size * num_heads,
                         pool_size, buffer_size)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.convs = nn.LayerList()
        for i in range(self.num_layers - 1):
            in_dim = input_size if i == 0 else hidden_size * num_heads
            conv = pgl.nn.GATConv(
                in_dim,
                hidden_size,
                feat_drop,
                attn_drop,
                num_heads,
                activation='elu')
            self.convs.append(conv)

        if self.num_layers == 1:
            hidden_dim = input_size
        else:
            hidden_dim = hidden_size * num_heads
        conv = pgl.nn.GATConv(
            hidden_dim,
            output_size,
            feat_drop,
            attn_drop,
            1,
            concat=False,
            activation=None)
        self.convs.append(conv)

    def forward(self, graph, x, norm, *args):
        for conv, hist in zip(self.convs[:-1], self.histories):
            x = conv(graph, x)
            x = self.push_and_pull(hist, x, *args)

        x = self.convs[-1](graph, x)
        return x

    @paddle.no_grad()
    def forward_layer(self, layer, graph, x, norm, state):
        x = self.convs[layer](graph, x)
        return x
