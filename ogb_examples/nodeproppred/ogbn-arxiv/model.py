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
# encoding=utf-8
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
import paddle.fluid as fluid
import paddle.fluid.layers as L
from pgl.graph_wrapper import GraphWrapper
from pgl.layers.conv import gcn, gat
from pgl.utils import paddle_helper


class BaseGraph(object):
    """Base Graph Model"""

    def __init__(self, args, graph_wrapper=None):
        self.hidden_size = args.hidden_size
        self.num_nodes = args.num_nodes
        self.drop_rate = args.drop_rate
        node_feature = [('feat', [None, 128], "float32")]
        if graph_wrapper is None:
            self.graph_wrapper = GraphWrapper(
                name="graph", place=F.CPUPlace(), node_feat=node_feature)
        else:
            self.graph_wrapper = graph_wrapper
        self.build_model(args)

    def build_model(self, args):
        """ build graph model"""
        self.batch_nodes = L.data(
            name="batch_nodes", shape=[-1], dtype="int64")
        self.labels = L.data(name="labels", shape=[-1], dtype="int64")

        self.batch_nodes = L.reshape(self.batch_nodes, [-1, 1])
        self.labels = L.reshape(self.labels, [-1, 1])

        self.batch_nodes.stop_gradients = True
        self.labels.stop_gradients = True

        feat = self.graph_wrapper.node_feat['feat']
        if self.graph_wrapper is not None:
            feat = self.neighbor_aggregator(feat)

        assert feat is not None
        feat = L.gather(feat, self.batch_nodes)
        self.logits = L.fc(feat,
                           size=40,
                           act=None,
                           name="node_predictor_logits")
        self.loss()

    def mlp(self, feat):
        for i in range(3):
            feat = L.fc(node,
                        size=self.hidden_size,
                        name="simple_mlp_{}".format(i))
            feat = L.batch_norm(feat)
            feat = L.relu(feat)
            feat = L.dropout(feat, dropout_prob=0.5)
        return feat

    def loss(self):
        self.loss = L.softmax_with_cross_entropy(self.logits, self.labels)
        self.loss = L.reduce_mean(self.loss)
        self.metrics = {"loss": self.loss, }

    def neighbor_aggregator(self, feature):
        """neighbor aggregation"""
        raise NotImplementedError(
            "Please implement this method when you using graph wrapper for GNNs."
        )


class MLPModel(BaseGraph):
    def __init__(self, args, gw):
        super(MLPModel, self).__init__(args, gw)

    def neighbor_aggregator(self, feature):
        for i in range(3):
            feature = L.fc(feature,
                           size=self.hidden_size,
                           name="simple_mlp_{}".format(i))
            #feature = L.batch_norm(feature)
            feature = L.relu(feature)
            feature = L.dropout(feature, dropout_prob=self.drop_rate)
        return feature


class SAGEModel(BaseGraph):
    def __init__(self, args, gw):
        super(SAGEModel, self).__init__(args, gw)

    def neighbor_aggregator(self, feature):
        sage = GraphSageModel(40, 3, 256)
        feature = sage.forward(self.graph_wrapper, feature, self.drop_rate)
        return feature


class GAANModel(BaseGraph):
    def __init__(self, args, gw):
        super(GAANModel, self).__init__(args, gw)

    def neighbor_aggregator(self, feature):
        gaan = GaANModel(
            40,
            3,
            hidden_size_a=48,
            hidden_size_v=64,
            hidden_size_m=128,
            hidden_size_o=256)
        feature = gaan.forward(self.graph_wrapper, feature, self.drop_rate)
        return feature


class GINModel(BaseGraph):
    def __init__(self, args, gw):
        super(GINModel, self).__init__(args, gw)

    def neighbor_aggregator(self, feature):
        gin = GinModel(40, 2, 256)
        feature = gin.forward(self.graph_wrapper, feature, self.drop_rate)
        return feature


class GATModel(BaseGraph):
    def __init__(self, args, gw):
        super(GATModel, self).__init__(args, gw)

    def neighbor_aggregator(self, feature):
        feature = gat(self.graph_wrapper,
                      feature,
                      hidden_size=self.hidden_size,
                      activation='relu',
                      name="GAT_1")
        feature = gat(self.graph_wrapper,
                      feature,
                      hidden_size=self.hidden_size,
                      activation='relu',
                      name="GAT_2")
        return feature


class GCNModel(BaseGraph):
    def __init__(self, args, gw):
        super(GCNModel, self).__init__(args, gw)

    def neighbor_aggregator(self, feature):
        feature = gcn(
            self.graph_wrapper,
            feature,
            hidden_size=self.hidden_size,
            activation='relu',
            name="GCN_1", )
        feature = fluid.layers.dropout(feature, dropout_prob=self.drop_rate)
        feature = gcn(self.graph_wrapper,
                      feature,
                      hidden_size=self.hidden_size,
                      activation='relu',
                      name="GCN_2")
        feature = fluid.layers.dropout(feature, dropout_prob=self.drop_rate)
        return feature


class GinModel(object):
    def __init__(self,
                 num_class,
                 num_layers,
                 hidden_size,
                 act='relu',
                 name="GINModel"):
        self.num_class = num_class
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.act = act
        self.name = name

    def forward(self, gw, feature):
        for i in range(self.num_layers):
            feature = gin(gw, feature, self.hidden_size, self.act,
                          self.name + '_' + str(i))
            feature = fluid.layers.layer_norm(
                feature,
                begin_norm_axis=1,
                param_attr=fluid.ParamAttr(
                    name="norm_scale_%s" % (i),
                    initializer=fluid.initializer.Constant(1.0)),
                bias_attr=fluid.ParamAttr(
                    name="norm_bias_%s" % (i),
                    initializer=fluid.initializer.Constant(0.0)), )

            feature = fluid.layers.relu(feature)
        return feature


class GaANModel(object):
    def __init__(self,
                 num_class,
                 num_layers,
                 hidden_size_a=24,
                 hidden_size_v=32,
                 hidden_size_m=64,
                 hidden_size_o=128,
                 heads=8,
                 act='relu',
                 name="GaAN"):
        self.num_class = num_class
        self.num_layers = num_layers
        self.hidden_size_a = hidden_size_a
        self.hidden_size_v = hidden_size_v
        self.hidden_size_m = hidden_size_m
        self.hidden_size_o = hidden_size_o
        self.act = act
        self.name = name
        self.heads = heads

    def GaANConv(self, gw, feature, name):
        feat_key = fluid.layers.fc(
            feature,
            self.hidden_size_a * self.heads,
            bias_attr=False,
            param_attr=fluid.ParamAttr(name=name + '_project_key'))
        # N * (D2 * M)
        feat_value = fluid.layers.fc(
            feature,
            self.hidden_size_v * self.heads,
            bias_attr=False,
            param_attr=fluid.ParamAttr(name=name + '_project_value'))
        # N * (D1 * M)
        feat_query = fluid.layers.fc(
            feature,
            self.hidden_size_a * self.heads,
            bias_attr=False,
            param_attr=fluid.ParamAttr(name=name + '_project_query'))
        # N * Dm
        feat_gate = fluid.layers.fc(
            feature,
            self.hidden_size_m,
            bias_attr=False,
            param_attr=fluid.ParamAttr(name=name + '_project_gate'))

        # send
        message = gw.send(
            self.send_func,
            nfeat_list=[('node_feat', feature), ('feat_key', feat_key),
                        ('feat_value', feat_value), ('feat_query', feat_query),
                        ('feat_gate', feat_gate)],
            efeat_list=None, )

        # recv
        output = gw.recv(message, self.recv_func)
        output = fluid.layers.fc(
            output,
            self.hidden_size_o,
            bias_attr=False,
            param_attr=fluid.ParamAttr(name=name + '_project_output'))
        output = fluid.layers.leaky_relu(output, alpha=0.1)
        output = fluid.layers.dropout(output, dropout_prob=0.1)
        return output

    def forward(self, gw, feature, drop_rate):
        for i in range(self.num_layers):
            feature = self.GaANConv(gw, feature, self.name + '_' + str(i))
            feature = fluid.layers.dropout(feature, dropout_prob=drop_rate)
        return feature

    def send_func(self, src_feat, dst_feat, edge_feat):
        # E * (M * D1)
        feat_query, feat_key = dst_feat['feat_query'], src_feat['feat_key']
        # E * M * D1
        old = feat_query
        feat_query = fluid.layers.reshape(
            feat_query, [-1, self.heads, self.hidden_size_a])
        feat_key = fluid.layers.reshape(feat_key,
                                        [-1, self.heads, self.hidden_size_a])
        # E * M
        alpha = fluid.layers.reduce_sum(feat_key * feat_query, dim=-1)

        return {
            'dst_node_feat': dst_feat['node_feat'],
            'src_node_feat': src_feat['node_feat'],
            'feat_value': src_feat['feat_value'],
            'alpha': alpha,
            'feat_gate': src_feat['feat_gate']
        }

    def recv_func(self, message):
        dst_feat = message['dst_node_feat']
        src_feat = message['src_node_feat']
        x = fluid.layers.sequence_pool(dst_feat, 'average')
        z = fluid.layers.sequence_pool(src_feat, 'average')

        feat_gate = message['feat_gate']
        g_max = fluid.layers.sequence_pool(feat_gate, 'max')

        g = fluid.layers.concat([x, g_max, z], axis=1)
        g = fluid.layers.fc(g, self.heads, bias_attr=False, act="sigmoid")

        # softmax
        alpha = message['alpha']
        alpha = paddle_helper.sequence_softmax(alpha)  # E * M

        feat_value = message['feat_value']  # E * (M * D2)
        old = feat_value
        feat_value = fluid.layers.reshape(
            feat_value, [-1, self.heads, self.hidden_size_v])  # E * M * D2
        feat_value = fluid.layers.elementwise_mul(feat_value, alpha, axis=0)
        feat_value = fluid.layers.reshape(
            feat_value, [-1, self.heads * self.hidden_size_v])  # E * (M * D2)
        feat_value = fluid.layers.lod_reset(feat_value, old)

        feat_value = fluid.layers.sequence_pool(feat_value,
                                                'sum')  # N * (M * D2)
        feat_value = fluid.layers.reshape(
            feat_value, [-1, self.heads, self.hidden_size_v])  # N * M * D2
        output = fluid.layers.elementwise_mul(feat_value, g, axis=0)
        output = fluid.layers.reshape(
            output, [-1, self.heads * self.hidden_size_v])  # N * (M * D2)
        output = fluid.layers.concat([x, output], axis=1)

        return output


class GraphSageModel(object):
    def __init__(self,
                 num_class,
                 num_layers,
                 hidden_size,
                 act='relu',
                 name="GraphSage"):
        self.num_class = num_class
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.act = act
        self.name = name

    def GraphSageConv(self, gw, feature, name):
        message = gw.send(
            self.send_func,
            nfeat_list=[('node_feat', feature)],
            efeat_list=None, )
        neighbor_feat = gw.recv(message, self.recv_func)
        neighbor_feat = fluid.layers.fc(neighbor_feat,
                                        self.hidden_size,
                                        act=self.act,
                                        name=name + '_n')
        self_feature = fluid.layers.fc(feature,
                                       self.hidden_size,
                                       act=self.act,
                                       name=name + '_s')
        output = self_feature + neighbor_feat
        output = fluid.layers.l2_normalize(output, axis=1)

        return output

    def SageConv(self, gw, feature, name, hidden_size, act):
        message = gw.send(
            self.send_func,
            nfeat_list=[('node_feat', feature)],
            efeat_list=None, )
        neighbor_feat = gw.recv(message, self.recv_func)
        neighbor_feat = fluid.layers.fc(neighbor_feat,
                                        hidden_size,
                                        act=None,
                                        name=name + '_n')
        self_feature = fluid.layers.fc(feature,
                                       hidden_size,
                                       act=None,
                                       name=name + '_s')
        output = self_feature + neighbor_feat
        # output = fluid.layers.concat([self_feature, neighbor_feat], axis=1)
        output = fluid.layers.l2_normalize(output, axis=1)
        if act is not None:
            ouput = L.relu(output)
        return output

    def bn_drop(self, feat, drop_rate):
        #feat = L.batch_norm(feat)
        feat = L.dropout(feat, dropout_prob=drop_rate)
        return feat

    def forward(self, gw, feature, drop_rate):
        for i in range(self.num_layers):
            final = (i == (self.num_layers - 1))
            feature = self.SageConv(gw, feature, self.name + '_' + str(i),
                                    self.hidden_size, None
                                    if final else self.act)
            if not final:
                feature = self.bn_drop(feature, drop_rate)
        return feature

    def send_func(self, src_feat, dst_feat, edge_feat):
        return src_feat["node_feat"]

    def recv_func(self, feat):
        return fluid.layers.sequence_pool(feat, pool_type="average")
