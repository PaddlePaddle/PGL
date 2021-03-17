# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import pgl
import paddle
import paddle.nn as nn


class RGCN(nn.Layer):
    """Implementation of R-GCN model.
    """

    def __init__(self, num_nodes, input_size, hidden_size, num_class,
                 num_layers, etypes, num_bases):
        super(RGCN, self).__init__()

        self.num_nodes = num_nodes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_class = num_class
        self.num_layers = num_layers
        self.etypes = etypes
        self.num_bases = num_bases

        self.nfeat = self.create_parameter([self.num_nodes, self.input_size])

        self.rgcns = nn.LayerList()
        out_dim = self.hidden_size
        for i in range(self.num_layers):
            in_dim = self.input_size if i == 0 else self.hidden_size
            self.rgcns.append(
                pgl.nn.RGCNConv(
                    in_dim,
                    out_dim,
                    self.etypes,
                    self.num_bases, ))

        self.linear = nn.Linear(self.hidden_size, self.num_class)

    def forward(self, g):
        h = self.nfeat
        for i in range(self.num_layers):
            h = self.rgcns[i](g, h)

        logits = self.linear(h)
        return logits
