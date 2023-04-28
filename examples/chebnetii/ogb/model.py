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
import math
from utils import cheby


class ChebNetII(nn.Layer):
    def __init__(self, num_features, hidden, num_classes, args):
        super(ChebNetII, self).__init__()
        self.lin1 = nn.Linear(num_features, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.lin3 = nn.Linear(hidden, num_classes)

        self.K = args.K
        self.temp = self.create_parameter(
            shape=[self.K + 1],
            dtype="float32",
            default_initializer=nn.initializer.Constant(value=1.0), )
        self.dropout = args.dropout
        self.feat_dropout = nn.Dropout(p=self.dropout)

    def forward(self, x_lis, st=0, end=0):
        coe_tmp = F.relu(self.temp)
        coe = coe_tmp.clone()

        for i in range(self.K + 1):
            coe[i] = coe_tmp[0] * cheby(i,
                                        math.cos((self.K + 0.5) * math.pi /
                                                 (self.K + 1)))
            for j in range(1, self.K + 1):
                x_j = math.cos((self.K - j + 0.5) * math.pi / (self.K + 1))
                coe[i] = coe[i] + coe_tmp[j] * cheby(i, x_j)
            coe[i] = 2 * coe[i] / (self.K + 1)

        Tx_0 = x_lis[0][st:end, :]
        out = coe[0] / 2 * Tx_0

        for k in range(1, self.K + 1):
            Tx_k = x_lis[k][st:end, :]
            out = out + coe[k] * Tx_k

        x = self.lin1(out)
        x = F.relu(x)
        x = self.feat_dropout(x)

        x = self.lin2(x)
        x = F.relu(x)
        x = self.feat_dropout(x)

        x = self.lin3(x)

        return x
