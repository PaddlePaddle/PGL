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


class DeepwalkModel(object):
    def __init__(self,
                 num_nodes,
                 hidden_size=16,
                 neg_num=5,
                 is_sparse=False,
                 is_distributed=False,
                 embedding_lr=1.0):
        self.pyreader = L.py_reader(
            capacity=70,
            shapes=[[-1, 1, 1], [-1, neg_num + 1, 1]],
            dtypes=['int64', 'int64'],
            lod_levels=[0, 0],
            name='train',
            use_double_buffer=True)

        self.num_nodes = num_nodes
        self.neg_num = neg_num

        self.embed_init = F.initializer.Uniform(
            low=-1. / math.sqrt(hidden_size), high=1. / math.sqrt(hidden_size))
        self.is_sparse = is_sparse
        self.is_distributed = is_distributed
        self.hidden_size = hidden_size
        self.loss = None
        self.embedding_lr = embedding_lr
        max_hidden_size = int(math.pow(2, 31) / 4 / num_nodes)
        self.num_part = int(math.ceil(1. * hidden_size / max_hidden_size))

    def forward(self):
        src, dsts = L.read_file(self.pyreader)

        if self.is_sparse:
            # sparse mode use 2 dims input.
            src = L.reshape(src, [-1, 1])
            dsts = L.reshape(dsts, [-1, 1])

        if self.num_part is not None and self.num_part != 1 and not self.is_distributed:
            src_embed = split_embedding(
                src,
                self.num_nodes,
                self.hidden_size,
                self.embed_init,
                "weight",
                self.num_part,
                self.is_sparse,
                learning_rate=self.embedding_lr)

            dsts_embed = split_embedding(
                dsts,
                self.num_nodes,
                self.hidden_size,
                self.embed_init,
                "weight",
                self.num_part,
                self.is_sparse,
                learning_rate=self.embedding_lr)
        else:
            src_embed = L.embedding(
                src, (self.num_nodes, self.hidden_size),
                self.is_sparse,
                self.is_distributed,
                param_attr=F.ParamAttr(
                    name="weight",
                    learning_rate=self.embedding_lr,
                    initializer=self.embed_init))

            dsts_embed = L.embedding(
                dsts, (self.num_nodes, self.hidden_size),
                self.is_sparse,
                self.is_distributed,
                param_attr=F.ParamAttr(
                    name="weight",
                    learning_rate=self.embedding_lr,
                    initializer=self.embed_init))

        if self.is_sparse:
            # reshape back
            src_embed = L.reshape(src_embed, [-1, 1, self.hidden_size])
            dsts_embed = L.reshape(dsts_embed,
                                   [-1, self.neg_num + 1, self.hidden_size])

        logits = L.matmul(
            src_embed, dsts_embed,
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
