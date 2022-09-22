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


class SageConv(BaseConv):
    """ GraphSAGE is a general inductive framework that leverages node feature
    information (e.g., text attributes) to efficiently generate node embeddings
    for previously unseen data.
    """

    def __init__(self, input_size, hidden_size, aggr_type="sum", act=None):
        super(SageConv, self).__init__()
        assert aggr_type in ["sum", "mean", "max", "min"], \
                "Only support 'sum', 'mean', 'max', 'min' built-in receive type."

        self.aggr_type = aggr_type
        self.self_linear = nn.Linear(input_size, hidden_size)
        self.neigh_linear = nn.Linear(input_size, hidden_size)

    def forward(self, edge_index, feature):
        neigh_feature = self.send_recv(edge_index, feature, self.aggr_type)
        self_feature = self.self_linear(feature)
        neigh_feature = self.neigh_linear(neigh_feature)
        output = self_feature + neigh_feature

        output = F.normalize(output, axis=1)
        return output


class GraphSage(nn.Layer):
    """Implementation of GraphSage.
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
                SageConv(input_size if i == 0 else hidden_size, hidden_size))

    def forward(self, edge_index, feature):
        for conv in self.convs:
            feature = conv(edge_index, feature)
        feature = self.linear(feature)
        return feature
