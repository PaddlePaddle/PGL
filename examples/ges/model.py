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
    GES model file.
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import math

import paddle.fluid.layers as L
import paddle.fluid as F


def split_embedding(input,
                    dict_size,
                    hidden_size,
                    initializer,
                    name,
                    num_part=16,
                    is_sparse=False,
                    learning_rate=1.0):
    """ split_embedding
    """
    _part_size = hidden_size // num_part
    if hidden_size % num_part != 0:
        _part_size += 1
    output_embedding = []
    p_num = 0
    while hidden_size > 0:
        _part_size = min(_part_size, hidden_size)
        hidden_size -= _part_size
        print("part", p_num, "size=", (dict_size, _part_size))
        part_embedding = L.embedding(
            input=input,
            size=(dict_size, _part_size),
            is_sparse=is_sparse,
            is_distributed=False,
            param_attr=F.ParamAttr(
                name=name + '_part%s' % p_num,
                initializer=initializer,
                learning_rate=learning_rate))
        p_num += 1
        output_embedding.append(part_embedding)
    return L.concat(output_embedding, -1)


class GESModel(object):
    """ GESModel
    """

    def __init__(self,
                 num_nodes,
                 num_featuers,
                 hidden_size=16,
                 neg_num=5,
                 is_sparse=False,
                 num_part=1):
        self.pyreader = L.py_reader(
            capacity=70,
            shapes=[[-1, 1, num_featuers, 1],
                    [-1, neg_num + 1, num_featuers, 1]],
            dtypes=['int64', 'int64'],
            lod_levels=[0, 0],
            name='train',
            use_double_buffer=True)

        self.num_nodes = num_nodes
        self.num_featuers = num_featuers
        self.neg_num = neg_num
        self.embed_init = F.initializer.TruncatedNormal(scale=1.0 /
                                                        math.sqrt(hidden_size))
        self.is_sparse = is_sparse
        self.num_part = num_part
        self.hidden_size = hidden_size
        self.loss = None

    def forward(self):
        """ forward
        """
        src, dst = L.read_file(self.pyreader)

        if self.is_sparse:
            # sparse mode use 2 dims input.
            src = L.reshape(src, [-1, 1])
            dst = L.reshape(dst, [-1, 1])

        src_embed = split_embedding(src, self.num_nodes, self.hidden_size,
                                    self.embed_init, "weight", self.num_part,
                                    self.is_sparse)

        dst_embed = split_embedding(dst, self.num_nodes, self.hidden_size,
                                    self.embed_init, "weight", self.num_part,
                                    self.is_sparse)

        if self.is_sparse:
            src_embed = L.reshape(
                src_embed, [-1, 1, self.num_featuers, self.hidden_size])
            dst_embed = L.reshape(
                dst_embed,
                [-1, self.neg_num + 1, self.num_featuers, self.hidden_size])

        src_embed = L.reduce_mean(src_embed, 2)
        dst_embed = L.reduce_mean(dst_embed, 2)

        logits = L.matmul(
            src_embed, dst_embed,
            transpose_y=True)  # [batch_size, 1, neg_num+1]

        pos_label = L.fill_constant_batch_size_like(logits, [-1, 1, 1],
                                                    "float32", 1)
        neg_label = L.fill_constant_batch_size_like(
            logits, [-1, 1, self.neg_num], "float32", 0)
        label = L.concat([pos_label, neg_label], -1)

        pos_weight = L.fill_constant_batch_size_like(logits, [-1, 1, 1],
                                                     "float32", self.neg_num)
        neg_weight = L.fill_constant_batch_size_like(
            logits, [-1, 1, self.neg_num], "float32", 1)
        weight = L.concat([pos_weight, neg_weight], -1)

        weight.stop_gradient = True
        label.stop_gradient = True

        loss = L.sigmoid_cross_entropy_with_logits(logits, label)
        loss = loss * weight
        loss = L.reduce_mean(loss)
        loss = loss * ((self.neg_num + 1) / 2 / self.neg_num)
        loss.persistable = True
        self.loss = loss
        return loss


class EGESModel(GESModel):
    """ EGESModel
    """

    def forward(self):
        """ forward
        """
        src, dst = L.read_file(self.pyreader)

        src_id = L.slice(src, [0, 1, 2, 3], [0, 0, 0, 0],
                         [int(math.pow(2, 30)) - 1, 1, 1, 1])
        dst_id = L.slice(dst, [0, 1, 2, 3], [0, 0, 0, 0],
                         [int(math.pow(2, 30)) - 1, self.neg_num + 1, 1, 1])

        if self.is_sparse:
            # sparse mode use 2 dims input.
            src = L.reshape(src, [-1, 1])
            dst = L.reshape(dst, [-1, 1])

        # [b, 1, f, h]
        src_embed = split_embedding(src, self.num_nodes, self.hidden_size,
                                    self.embed_init, "weight", self.num_part,
                                    self.is_sparse)

        # [b, n+1, f, h]
        dst_embed = split_embedding(dst, self.num_nodes, self.hidden_size,
                                    self.embed_init, "weight", self.num_part,
                                    self.is_sparse)

        if self.is_sparse:
            src_embed = L.reshape(
                src_embed, [-1, 1, self.num_featuers, self.hidden_size])
            dst_embed = L.reshape(
                dst_embed,
                [-1, self.neg_num + 1, self.num_featuers, self.hidden_size])

        # [b, 1, 1, f]
        src_weight = L.softmax(
            L.embedding(
                src_id, [self.num_nodes, self.num_featuers],
                param_attr=F.ParamAttr(name="alpha")))
        # [b, n+1, 1, f]
        dst_weight = L.softmax(
            L.embedding(
                dst_id, [self.num_nodes, self.num_featuers],
                param_attr=F.ParamAttr(name="alpha")))

        # [b, 1, h]
        src_sum = L.squeeze(L.matmul(src_weight, src_embed), axes=[2])
        # [b, n+1, h]
        dst_sum = L.squeeze(L.matmul(dst_weight, dst_embed), axes=[2])

        logits = L.matmul(
            src_sum, dst_sum, transpose_y=True)  # [batch_size, 1, neg_num+1]

        pos_label = L.fill_constant_batch_size_like(logits, [-1, 1, 1],
                                                    "float32", 1)
        neg_label = L.fill_constant_batch_size_like(
            logits, [-1, 1, self.neg_num], "float32", 0)
        label = L.concat([pos_label, neg_label], -1)

        pos_weight = L.fill_constant_batch_size_like(logits, [-1, 1, 1],
                                                     "float32", self.neg_num)
        neg_weight = L.fill_constant_batch_size_like(
            logits, [-1, 1, self.neg_num], "float32", 1)
        weight = L.concat([pos_weight, neg_weight], -1)

        weight.stop_gradient = True
        label.stop_gradient = True

        loss = L.sigmoid_cross_entropy_with_logits(logits, label)
        loss = loss * weight
        loss = L.reduce_mean(loss)
        loss = loss * ((self.neg_num + 1) / 2 / self.neg_num)
        loss.persistable = True
        self.loss = loss
        return loss
