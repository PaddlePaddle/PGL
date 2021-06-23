# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
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
"""Model Definition
"""
import math
import time

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

__all__ = ["SkipGramModel"]


class SparseEmbedding(object):
    def __init__(self, num, dim, weight_attr=None):
        self.num = num
        self.dim = dim
        self.emb_attr = weight_attr

    def forward(self, x):
        x_shape = paddle.shape(x)
        x = paddle.reshape(x, [-1, 1])

        x_emb = paddle.static.nn.sparse_embedding(
            x, [self.num, self.dim], param_attr=self.emb_attr)

        return paddle.reshape(x_emb, [x_shape[0], x_shape[1], self.dim])

    def __call__(self, x):
        return self.forward(x)


class SkipGramModel(nn.Layer):
    def __init__(self, config):
        super(SkipGramModel, self).__init__()

        self.config = config
        self.embed_size = self.config.embed_size
        self.neg_num = self.config.neg_num

        embed_init = nn.initializer.Uniform(
            low=-1. / math.sqrt(self.embed_size),
            high=1. / math.sqrt(self.embed_size))
        emb_attr = paddle.ParamAttr(
            name="node_embedding", initializer=embed_init)

        if config.sparse_embed:
            self.embedding = SparseEmbedding(
                self.config.num_nodes, self.embed_size, weight_attr=emb_attr)
        else:
            self.embedding = nn.Embedding(
                self.config.num_nodes, self.embed_size, weight_attr=emb_attr)

        self.loss_fn = paddle.nn.BCEWithLogitsLoss()

    def forward(self, feed_dict):
        src_embed = self.embedding(feed_dict['src'])
        pos_embed = self.embedding(feed_dict['pos'])

        # batch neg sample
        batch_size = feed_dict['pos'].shape[0]
        neg_idx = paddle.randint(
            low=0, high=batch_size, shape=[batch_size, self.neg_num])
        negs = []
        for i in range(self.neg_num):
            tmp = paddle.gather(pos_embed, neg_idx[:, i])
            tmp = paddle.reshape(tmp, [-1, 1, self.embed_size])
            negs.append(tmp)

        neg_embed = paddle.concat(negs, axis=1)
        src_embed = paddle.reshape(src_embed, [-1, 1, self.embed_size])
        pos_embed = paddle.reshape(pos_embed, [-1, 1, self.embed_size])

        # [batch_size, 1, 1]
        pos_logits = paddle.matmul(src_embed, pos_embed, transpose_y=True)
        # [batch_size, 1, neg_num]
        neg_logits = paddle.matmul(src_embed, neg_embed, transpose_y=True)

        ones_label = paddle.ones_like(pos_logits)
        pos_loss = self.loss_fn(pos_logits, ones_label)

        zeros_label = paddle.zeros_like(neg_logits)
        neg_loss = self.loss_fn(neg_logits, zeros_label)

        loss = (pos_loss + neg_loss) / 2
        return loss
