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
"""This file implement the GIN model.
"""

import numpy as np

import paddle.fluid as fluid
import paddle.fluid.layers as fl
import pgl
from pgl.layers.conv import gin


class GINModel(object):
    """GINModel"""

    def __init__(self, args, gw, num_class):
        self.args = args
        self.num_layers = self.args.num_layers
        self.hidden_size = self.args.hidden_size
        self.train_eps = self.args.train_eps
        self.pool_type = self.args.pool_type
        self.dropout_prob = self.args.dropout_prob
        self.num_class = num_class

        self.gw = gw
        self.labels = fl.data(name="labels", shape=[None, 1], dtype="int64")

    def forward(self):
        """forward"""
        features_list = [self.gw.node_feat["attr"]]

        for i in range(self.num_layers):
            h = gin(self.gw,
                    features_list[i],
                    hidden_size=self.hidden_size,
                    activation="relu",
                    name="gin_%s" % (i),
                    init_eps=0.0,
                    train_eps=self.train_eps)

            h = fl.layer_norm(
                h,
                begin_norm_axis=1,
                param_attr=fluid.ParamAttr(
                    name="norm_scale_%s" % (i),
                    initializer=fluid.initializer.Constant(1.0)),
                bias_attr=fluid.ParamAttr(
                    name="norm_bias_%s" % (i),
                    initializer=fluid.initializer.Constant(0.0)), )

            h = fl.relu(h)

            features_list.append(h)

        output = 0
        for i, h in enumerate(features_list):
            pooled_h = pgl.layers.graph_pooling(self.gw, h, self.pool_type)
            drop_h = fl.dropout(
                pooled_h,
                self.dropout_prob,
                dropout_implementation="upscale_in_train")
            output += fl.fc(drop_h,
                            size=self.num_class,
                            act=None,
                            param_attr=fluid.ParamAttr(name="final_fc_%s" %
                                                       (i)))

        # calculate loss
        self.loss = fl.softmax_with_cross_entropy(output, self.labels)
        self.loss = fl.reduce_mean(self.loss)
        self.acc = fl.accuracy(fl.softmax(output), self.labels)
