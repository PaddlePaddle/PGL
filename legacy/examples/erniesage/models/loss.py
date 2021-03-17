# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
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
import time
import glob
import os

import numpy as np

import pgl
import paddle.fluid as F
import paddle.fluid.layers as L


class Loss(object):

    def __init__(self, config):
        self.config = config

    @classmethod
    def factory(cls, config):
        loss_type = config.loss_type
        if loss_type == "hinge":
            return HingeLoss(config)
        elif loss_type == "global_hinge":
            return GlobalHingeLoss(config)
        elif loss_type == "softmax_with_cross_entropy":
            return lambda logits, label: L.reduce_mean(L.softmax_with_cross_entropy(logits, label))
        else:
            raise ValueError


class HingeLoss(Loss):

    def __call__(self, user_feat, pos_item_feat, neg_item_feat):
        pos = L.reduce_sum(user_feat * pos_item_feat, -1, keep_dim=True) # [B, 1]
        neg = L.matmul(user_feat, neg_item_feat, transpose_y=True) # [B, B]
        loss = L.reduce_mean(L.relu(neg - pos + self.config.margin))
        return loss


def all_gather(X):
    trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
    trainer_num = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
    if trainer_num == 1:
        copy_X = X * 1
        copy_X.stop_gradient=True
        return copy_X

    Xs = []
    for i in range(trainer_num):
        copy_X = X * 1
        copy_X =  L.collective._broadcast(copy_X, i, True)
        copy_X.stop_gradient=True
        Xs.append(copy_X)

    if len(Xs) > 1:
        Xs=L.concat(Xs, 0)
        Xs.stop_gradient=True
    else:
        Xs = Xs[0]
    return Xs


class GlobalHingeLoss(Loss):

    def __call__(self, user_feat, pos_item_feat, neg_item_feat):
        pos = L.reduce_sum(user_feat * pos_item_feat, -1, keep_dim=True) # [B, 1]
        all_pos = all_gather(pos) # [B * n, 1]
        all_neg_item_feat = all_gather(neg_item_feat) # [B * n, 1]
        all_user_feat = all_gather(user_feat) # [B * n, 1]

        neg1 = L.matmul(user_feat, all_neg_item_feat, transpose_y=True) # [B, B * n]
        neg2 = L.matmul(all_user_feat, neg_item_feat, transpose_y=True) # [B *n, B]

        loss1 = L.reduce_mean(L.relu(neg1 - pos + self.config.margin))
        loss2 = L.reduce_mean(L.relu(neg2 - all_pos + self.config.margin))

        loss = loss1 + loss2
        return loss
