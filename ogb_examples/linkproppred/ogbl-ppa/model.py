# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""lbs_model"""
import os
import re
import time
from random import random
from functools import reduce, partial

import numpy as np
import multiprocessing

import paddle
import paddle.fluid as F
import paddle.fluid.layers as L
from pgl.graph_wrapper import GraphWrapper
from pgl.layers.conv import gcn, gat


class BaseGraph(object):
    """Base Graph Model"""

    def __init__(self, args):
        node_feature = [('nfeat', [None, 58], "float32"),
                        ('node_id', [None, 1], "int64")]
        self.hidden_size = args.hidden_size
        self.num_nodes = args.num_nodes

        self.graph_wrapper = None  # GraphWrapper(
        #name="graph", place=F.CPUPlace(), node_feat=node_feature)

        self.build_model(args)

    def build_model(self, args):
        """ build graph model"""
        self.batch_src = L.data(name="batch_src", shape=[-1], dtype="int64")
        self.batch_src = L.reshape(self.batch_src, [-1, 1])
        self.batch_dst = L.data(name="batch_dst", shape=[-1], dtype="int64")
        self.batch_dst = L.reshape(self.batch_dst, [-1, 1])
        self.labels = L.data(name="labels", shape=[-1], dtype="int64")
        self.labels = L.reshape(self.labels, [-1, 1])
        self.labels.stop_gradients = True
        self.src_repr = L.embedding(
            self.batch_src,
            size=(self.num_nodes, self.hidden_size),
            param_attr=F.ParamAttr(
                name="node_embeddings",
                initializer=F.initializer.NormalInitializer(
                    loc=0.0, scale=1.0)))

        self.dst_repr = L.embedding(
            self.batch_dst,
            size=(self.num_nodes, self.hidden_size),
            param_attr=F.ParamAttr(
                name="node_embeddings",
                initializer=F.initializer.NormalInitializer(
                    loc=0.0, scale=1.0)))

        self.link_predictor(self.src_repr, self.dst_repr)

        self.bce_loss()

    def link_predictor(self, x, y):
        """ siamese network"""
        feat = x * y

        feat = L.fc(feat, size=self.hidden_size, name="link_predictor_1")
        feat = L.relu(feat)

        feat = L.fc(feat, size=self.hidden_size, name="link_predictor_2")
        feat = L.relu(feat)

        self.logits = L.fc(feat,
                           size=1,
                           act="sigmoid",
                           name="link_predictor_logits")

    def bce_loss(self):
        """listwise model"""
        mask = L.cast(self.labels > 0.5, dtype="float32")
        mask.stop_gradients = True

        self.loss = L.log_loss(self.logits, mask, epsilon=1e-15)
        self.loss = L.reduce_mean(self.loss) * 2
        proba = L.sigmoid(self.logits)
        proba = L.concat([proba * -1 + 1, proba], axis=1)
        auc_out, batch_auc_out, _ = \
             L.auc(input=proba, label=self.labels, curve='ROC', slide_steps=1)

        self.metrics = {
            "loss": self.loss,
            "auc": batch_auc_out,
        }

    def neighbor_aggregator(self, node_repr):
        """neighbor aggregation"""
        return node_repr
