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
This file implement the skipgram model for training metapath2vec.
"""

import argparse
import time
import math
import os
import io
from multiprocessing import Pool
import logging
import numpy as np
import glob

import pgl
from pgl import data_loader
from pgl.utils import op
from pgl.utils.logger import log
import paddle.fluid as fluid
import paddle.fluid.layers as fl


class SkipgramModel(object):
    """Implemetation of skipgram model.

    Args:
        config: dict, some configure parameters.
        dataset: instance of Dataset class
        place: GPU or CPU place
    """

    def __init__(self, config, dataset, place):
        self.config = config
        self.dataset = dataset
        self.place = place
        self.neg_num = self.dataset.config['neg_num']

        self.num_nodes = len(dataset.word2id)

        self.train_inputs = fl.data(
            'train_inputs', shape=[None, 1, 1], dtype='int64')

        self.train_labels = fl.data(
            'train_labels', shape=[None, 1, 1], dtype='int64')

        self.train_negs = fl.data(
            'train_negs', shape=[None, self.neg_num, 1], dtype='int64')

        self.forward()

    def backward(self, global_steps, opt_config):
        """Build the optimizer.
        """
        self.lr = fl.polynomial_decay(opt_config['lr'], global_steps,
                                      opt_config['end_lr'])
        adam = fluid.optimizer.Adam(learning_rate=self.lr)
        adam.minimize(self.loss)

    def forward(self):
        """Build the skipgram model.
        """
        initrange = 1.0 / self.config['embed_dim']
        embed_init = fluid.initializer.UniformInitializer(
            low=-initrange, high=initrange)
        weight_init = fluid.initializer.TruncatedNormal(
            scale=1.0 / math.sqrt(self.config['embed_dim']))

        embed_src = fl.embedding(
            input=self.train_inputs,
            size=[self.num_nodes, self.config['embed_dim']],
            param_attr=fluid.ParamAttr(
                name='content', initializer=embed_init))

        weight_pos = fl.embedding(
            input=self.train_labels,
            size=[self.num_nodes, self.config['embed_dim']],
            param_attr=fluid.ParamAttr(
                name='weight', initializer=weight_init))

        weight_negs = fl.embedding(
            input=self.train_negs,
            size=[self.num_nodes, self.config['embed_dim']],
            param_attr=fluid.ParamAttr(
                name='weight', initializer=weight_init))

        pos_logits = fl.matmul(
            embed_src, weight_pos, transpose_y=True)  # [batch_size, 1, 1]

        pos_score = fl.squeeze(pos_logits, axes=[1])
        pos_score = fl.clip(pos_score, min=-10, max=10)
        pos_score = -self.neg_num * fl.logsigmoid(pos_score)

        neg_logits = fl.matmul(
            embed_src, weight_negs,
            transpose_y=True)  # [batch_size, 1, neg_num]
        neg_score = fl.squeeze(neg_logits, axes=[1])
        neg_score = fl.clip(neg_score, min=-10, max=10)
        neg_score = -1.0 * fl.logsigmoid(-1.0 * neg_score)
        neg_score = fl.reduce_sum(neg_score, dim=1, keep_dim=True)

        self.loss = fl.reduce_mean(pos_score + neg_score) / self.neg_num / 2
