# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn.functional as F
from propagation import ChebProp


class ChebNetII(nn.Layer):
    """Implement of ChebNetII"""

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_class,
                 K=10,
                 drop=0.5,
                 dprate=0.5,
                 **kwargs):
        super(ChebNetII, self).__init__()
        self.K = K
        self.drop = drop
        self.dprate = dprate

        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, num_class)
        self.feat_dropout_1 = nn.Dropout(p=drop)
        self.feat_dropout_2 = nn.Dropout(p=dprate)

        self.prop = ChebProp(K=self.K)

    def forward(self, graph, feature):
        feature = self.feat_dropout_1(feature)
        feature = F.relu(self.linear_1(feature))

        feature = self.feat_dropout_1(feature)
        feature = self.linear_2(feature)

        if self.dprate > 0.0:
            feature = self.feat_dropout_2(feature)
        feature = self.prop(graph, feature)

        return feature
