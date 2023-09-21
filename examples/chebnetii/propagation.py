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

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import pgl
from pgl.nn import functional as GF
import math
from utils import cheby


class ChebProp(nn.Layer):
    def __init__(
            self,
            K=10, ):
        super(ChebProp, self).__init__()
        self.K = K
        self.temp = self.create_parameter(
            shape=[self.K + 1],
            dtype="float32",
            default_initializer=nn.initializer.Constant(value=1.0), )

    def forward(self, graph, feature, norm=None):
        coe_tmp = F.relu(self.temp)
        coe = coe_tmp.clone()

        norm = GF.degree_norm(graph)
        for i in range(self.K + 1):
            coe[i] = coe_tmp[0] * cheby(i,
                                        math.cos((self.K + 0.5) * math.pi /
                                                 (self.K + 1)))
            for j in range(1, self.K + 1):
                x_j = math.cos((self.K - j + 0.5) * math.pi / (self.K + 1))
                coe[i] = coe[i] + coe_tmp[j] * cheby(i, x_j)
            coe[i] = 2 * coe[i] / (self.K + 1)

        Tx_0 = feature
        feature = Tx_0 * norm
        feature = graph.send_recv(feature, reduce_func="sum")
        Tx_1 = feature * norm

        out = coe[0] / 2 * Tx_0 + coe[1] * Tx_1

        for i in range(2, self.K + 1):
            feature = Tx_1 * norm
            feature = graph.send_recv(feature, reduce_func="sum")
            Tx_2 = feature * norm

            Tx_2 = 2 * Tx_2 - Tx_0
            out = out + coe[i] * Tx_2
            Tx_0, Tx_1 = Tx_1, Tx_2
        return out
