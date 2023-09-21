# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved
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


class SageConv(BaseConv):
    def __init__(self, input_size, hidden_size, aggr_type="sum"):
        super(GraphSageConv, self).__init__()
        assert aggr_type in ["sum", "mean", "max", "min"], \
                "Only support 'sum', 'mean', 'max', 'min' built-in receive type."

        self.aggr_type = aggr_type
        self.self_linear = nn.Linear(input_size, hidden_size)
        self.neigh_linear = nn.Linear(input_size, hidden_size)

    def forward(self, edge_index, num_nodes, feature, act=None):
        neigh_feature = self.send_recv(feature, self.aggr_type)
        self_feature = self.self_linear(feature)
        neigh_feature = self.neigh_linear(neigh_feature)
        output = self_feature + neigh_feature
        if act is not None:
            output = getattr(F, act)(output)

        output = F.normalize(output, axis=1)
        return output
