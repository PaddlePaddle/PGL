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
from scipy.special import comb


class BernProp(nn.Layer):
    def __init__(
            self,
            k=10, ):
        super(BernProp, self).__init__()
        self.K = K
        self.temp = self.create_parameter(
            shape=[self.K + 1],
            dtype="float32",
            default_initializer=nn.initializer.Constant(value=1.0), )

    def forward(self, graph, feature, norm=None):
        TEMP = F.relu(self.temp)
        norm = GF.degree_norm(graph)
        tmp = []
        tmp.append(feature)
        for i in range(self.K):
            h0 = feature
            feature = feature * norm
            feature = graph.send_recv(feature)
            feature = feature * norm
            feature = h0 + feature
            tmp.append(feature)
        out = (comb(self.K, 0) / (2**self.K)) * TEMP[0] * tmp[self.K]
        for i in range(self.K):
            feature = tmp[self.K - i - 1]
            h0 = feature
            feature = feature * norm
            feature = graph.send_recv(feature)
            feature = feature * norm
            feature = h0 - feature
            for j in range(i):
                h0 = feature
                feature = feature * norm
                feature = graph.send_recv(feature)
                feature = feature * norm
                feature = h0 - feature
            out = out + (comb(self.K, i + 1) /
                         (2**self.K)) * TEMP[i + 1] * feature
        return out
