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
import paddle.nn as nn


class GraphSage(nn.Layer):
    """Implement of GraphSage
    """

    def __init__(self,
                 input_size,
                 num_class,
                 num_layers=1,
                 hidden_size=64,
                 dropout=0.5,
                 **kwargs):
        super(GraphSage, self).__init__()
        self.num_class = num_class
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.convs = nn.LayerList()
        self.linear = nn.Linear(self.hidden_size, self.num_class)
        for i in range(self.num_layers):
            self.convs.append(
                pgl.nn.GraphSageConv(input_size
                                     if i == 0 else hidden_size, hidden_size))

    def forward(self, graph, feature):
        for conv in self.convs:
            feature = conv(graph, feature)
        feature = self.linear(feature)
        return feature
