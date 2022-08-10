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
"""This file implement the GIN model + SAGPool.
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
from pgl.nn import SAGPool, GINConv
from pgl.math import segment_max


class GINModel(nn.Layer):
    """Implementation of GIN + SAGPool.
    """

    def __init__(self, args, num_class):
        super(GINModel, self).__init__()
        self.args = args
        self.input_size = self.args.feat_size
        self.output_size = self.args.hidden_size
        self.init_eps = self.args.init_eps
        self.train_eps = self.args.train_eps
        self.ratio = self.args.pool_ratio
        self.num_class = num_class
        self.min_score = self.args.min_score
        self.conv1 = pgl.nn.GINConv(self.input_size, self.output_size, "relu",
                                    self.init_eps, self.train_eps)
        self.pool1 = SAGPool(
            self.output_size,
            min_score=self.min_score,
            ratio=self.ratio,
            gnn=pgl.nn.GCNConv)
        self.conv2 = pgl.nn.GINConv(self.output_size, self.output_size, "relu",
                                    self.init_eps, self.train_eps)
        self.pool2 = SAGPool(
            self.output_size,
            min_score=self.min_score,
            ratio=self.ratio,
            gnn=pgl.nn.GCNConv)
        self.conv3 = pgl.nn.GINConv(self.output_size, self.output_size, "relu",
                                    self.init_eps, self.train_eps)
        self.out_layer = nn.Linear(self.output_size, self.num_class)

    def forward(self, graph):
        """
        SAGPool Forward
        """
        x = graph.node_feat["attr"]
        g = graph
        x = F.relu(self.conv1(g, x))
        x, _, g = self.pool1(g, x)
        x = F.relu(self.conv2(g, x))
        x, _, g = self.pool2(g, x)
        x = F.relu(self.conv3(g, x))
        x = segment_max(x, g.graph_node_id)
        x = self.out_layer(x)
        return x
