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
"""
    Deepwalk model file.
"""
import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class SkipGramModel(nn.Layer):
    def __init__(self, num_nodes, embed_size=16, neg_num=5, sparse=False):
        super(SkipGramModel, self).__init__()

        self.num_nodes = num_nodes
        self.neg_num = neg_num

        embed_init = nn.initializer.Uniform(
            low=-1. / math.sqrt(embed_size), high=1. / math.sqrt(embed_size))
        emb_attr = paddle.ParamAttr(name="node_embedding")
        self.emb = nn.Embedding(
            num_nodes, embed_size, sparse=sparse, weight_attr=emb_attr)
        self.loss = paddle.nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, src, dsts):
        # src [b, 1]
        # dsts [b, 1+neg]

        src_embed = self.emb(src)
        dsts_embed = self.emb(dsts)

        logits = paddle.matmul(
            src_embed, dsts_embed,
            transpose_y=True)  # [batch_size, 1, neg_num+1]

        dst_shape = paddle.shape(dsts)
        batch_size = dst_shape[0]
        neg_num = dst_shape[1] - 1

        pos_label = paddle.ones([batch_size, 1, 1], "float32")
        neg_label = paddle.zeros([batch_size, 1, neg_num], "float32")
        label = paddle.concat([pos_label, neg_label], -1)

        pos_weight = pos_label * neg_num
        neg_weight = neg_label + 1
        weight = paddle.concat([pos_weight, neg_weight], -1)

        #return logits, label, weight
        loss = self.loss(logits, label)
        loss = loss * weight
        loss = paddle.mean(loss)
        loss = loss * ((neg_num + 1) / 2 / neg_num)
        return loss
