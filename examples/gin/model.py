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
"""This file implement the GIN model.
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


class GINModel(nn.Layer):
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
        self.num_class = num_class

        self.gin_convs = nn.LayerList()
        self.norms = nn.LayerList()
        self.linears = nn.LayerList()
        self.linears.append(nn.Linear(self.input_size, self.num_class))
        for i in range(self.num_layers):
            if i == 0:
                input_size = self.input_size
            else:
                input_size = self.output_size
            gin = pgl.nn.GINConv(input_size, self.output_size, "relu",
                                 self.init_eps, self.train_eps)
            self.gin_convs.append(gin)
            ln = paddle.nn.LayerNorm(self.output_size)
            self.norms.append(ln)
            self.linears.append(nn.Linear(self.output_size, self.num_class))

        self.relu = nn.ReLU()
        self.graph_pooling = pgl.nn.GraphPool(self.pool_type)

    def forward(self, graph):
        features_list = [graph.node_feat['attr']]
        for i in range(self.num_layers):
            h = self.gin_convs[i](graph, features_list[i])
            h = self.norms[i](h)
            h = self.relu(h)
            features_list.append(h)

        output = 0
        for i, h in enumerate(features_list):
            h = self.graph_pooling(graph, h)
            h = F.dropout(h, p=self.dropout_prob)
            output += self.linears[i](h)

        return output
