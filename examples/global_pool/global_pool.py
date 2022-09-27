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
"""This file implement the GIN model + Graph Global Pooling.
"""

import os
import sys
import time
import argparse
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import pgl
from pgl.nn import GlobalAttention, Set2Set, GraphMultisetTransformer


class GINModel(nn.Layer):
    """Implementation of GIN model + Global Pool(mean, GMT, GlobalAttention, Set2Set).
    """

    def __init__(self, args, num_class):
        super(GINModel, self).__init__()
        self.args = args
        self.num_layers = self.args.num_layers
        self.input_size = self.args.feat_size
        self.output_size = self.args.hidden_size
        self.init_eps = self.args.init_eps
        self.train_eps = self.args.train_eps
        self.pool_type = self.args.pool_type
        self.dropout_prob = self.args.dropout_prob
        self.use_residual = self.args.use_residual
        self.num_nodes = self.args.num_nodes
        self.num_class = num_class
        self.jk = self.args.jk
        self.gin_convs = nn.LayerList()
        self.norms = nn.LayerList()
        self.linears = nn.LayerList()
        self.final_dim = self.output_size * 2 if self.pool_type == "Set2Set" else self.output_size
        self.out_layer = nn.Linear(self.final_dim, self.num_class)
        for i in range(self.num_layers):
            if i == 0:
                input_size = self.input_size
            else:
                input_size = self.output_size
            gin = pgl.nn.GINConv(input_size, self.output_size, "relu",
                                 self.init_eps, self.train_eps)
            self.gin_convs.append(gin)
            ln = paddle.nn.BatchNorm1D(self.output_size)
            self.norms.append(ln)

        self.relu = nn.ReLU()
        if self.pool_type in ["sum", "mean", "max"]:
            self.graph_pooling = pgl.nn.GraphPool(self.pool_type)
        elif self.pool_type == "GlobalAttention":
            self.graph_pooling = GlobalAttention(
                nn.Linear(self.output_size, 1))
        elif self.pool_type == "Set2Set":
            self.graph_pooling = Set2Set(self.output_size, n_iters=2)
        elif self.pool_type == "GMT":
            self.graph_pooling = GraphMultisetTransformer(
                self.output_size,
                self.output_size,
                self.output_size,
                num_nodes=self.num_nodes)
        else:
            raise NotImplementedError

    def forward(self, graph):
        """
        Forward
        """
        features_list = [graph.node_feat['attr']]
        for i in range(self.num_layers):
            h = self.gin_convs[i](graph, features_list[i])
            h = self.norms[i](h)
            if self.use_residual and i > 0:
                h = h + features_list[-1]
            h = F.dropout(
                self.relu(h), p=self.dropout_prob, training=self.training)
            features_list.append(h)
        # Jump Knowledge
        if self.jk == "sum":
            h = 0
            for i in features_list[1:]:
                h += i
        elif self.jk == "last":
            h = features_list[-1]
        output = self.graph_pooling(graph, h)
        return self.out_layer(output)
