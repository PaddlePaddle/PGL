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

import os

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

__all__ = ["MSELoss", "HuberLoss", "MAELoss", "SmoothMSELoss"]


class MSELoss(nn.Layer):
    def __init__(self, **kwargs):
        super(MSELoss, self).__init__()

    def forward(self, pred, gold, raw, col_names):
        # Remove bad input
        return F.mse_loss(pred, gold)


class MAELoss(nn.Layer):
    def __init__(self, **kwargs):
        super(MAELoss, self).__init__()

    def forward(self, pred, gold, raw, col_names):
        loss = F.l1_loss(pred, gold)
        return loss


class HuberLoss(nn.Layer):
    def __init__(self, delta=5, **kwargs):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, pred, gold, raw, col_names):
        loss = F.smooth_l1_loss(pred, gold, reduction='mean', delta=self.delta)
        return loss


class SmoothMSELoss(nn.Layer):
    def __init__(self, **kwargs):
        super(SmoothMSELoss, self).__init__()
        self.smooth_win = kwargs["smooth_win"]

    def forward(self, pred, gold, raw, col_names):
        gold = F.avg_pool1d(
            gold, self.smooth_win, stride=1, padding="SAME", exclusive=False)
        loss = F.mse_loss(pred, gold)
        return loss
