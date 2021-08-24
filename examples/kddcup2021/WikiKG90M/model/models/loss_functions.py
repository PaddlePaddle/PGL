# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved
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
"""
Loss functions.
"""

import paddle
import paddle.fluid as fluid
from paddle.nn.functional import log_sigmoid, sigmoid


class HingeLoss(object):
    def __init__(self, margin):
        super(HingeLoss, self).__init__()
        self.margin = margin

    def __call__(self, score, label):
        loss = self.margin - label * score
        loss = fluid.layers.relu(loss)
        return loss


class LogSigmoidLoss(object):
    def __init__(self):
        super(LogSigmoidLoss, self).__init__()

    def __call__(self, score, label):
        return -log_sigmoid(label * score)


class BCELoss(object):
    def __init__(self):
        super(BCELoss, self).__init__()

    def __call__(self, score, label):
        return -(label * paddle.log(sigmoid(score)) +
                 (1 - label) * paddle.log(1 - sigmoid(score)))
