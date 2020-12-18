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
    graphsage model.
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import math

import pgl
import numpy as np
import paddle
import paddle.fluid.layers as L
import paddle.fluid as F
import paddle.fluid as fluid

def copy_send(src_feat, dst_feat, edge_feat):
    return src_feat["h"]


def mean_recv(feat):
    return fluid.layers.sequence_pool(feat, pool_type="average")


def sum_recv(feat):
    return fluid.layers.sequence_pool(feat, pool_type="sum")


def max_recv(feat):
    return fluid.layers.sequence_pool(feat, pool_type="max")


def lstm_recv(feat):
    hidden_dim = 128
    forward, _ = fluid.layers.dynamic_lstm(
        input=feat, size=hidden_dim * 4, use_peepholes=False)
    output = fluid.layers.sequence_last_step(forward)
    return output


def graphsage_mean(gw, feature, hidden_size, act, name):
    msg = gw.send(copy_send, nfeat_list=[("h", feature)])
    neigh_feature = gw.recv(msg, mean_recv)
    self_feature = feature
    self_feature = fluid.layers.fc(self_feature,
                                   hidden_size,
                                   act=act,
                                   name=name + '_l')
    neigh_feature = fluid.layers.fc(neigh_feature,
                                    hidden_size,
                                    act=act,
                                    name=name + '_r')
    output = fluid.layers.concat([self_feature, neigh_feature], axis=1)
    output = fluid.layers.l2_normalize(output, axis=1)
    return output


def graphsage_meanpool(gw,
                       feature,
                       hidden_size,
                       act,
                       name,
                       inner_hidden_size=512):
    neigh_feature = fluid.layers.fc(feature, inner_hidden_size, act="relu")
    msg = gw.send(copy_send, nfeat_list=[("h", neigh_feature)])
    neigh_feature = gw.recv(msg, mean_recv)
    neigh_feature = fluid.layers.fc(neigh_feature,
                                    hidden_size,
                                    act=act,
                                    name=name + '_r')

    self_feature = feature
    self_feature = fluid.layers.fc(self_feature,
                                   hidden_size,
                                   act=act,
                                   name=name + '_l')
    output = fluid.layers.concat([self_feature, neigh_feature], axis=1)
    output = fluid.layers.l2_normalize(output, axis=1)
    return output


def graphsage_maxpool(gw,
                      feature,
                      hidden_size,
                      act,
                      name,
                      inner_hidden_size=512):
    neigh_feature = fluid.layers.fc(feature, inner_hidden_size, act="relu")
    msg = gw.send(copy_send, nfeat_list=[("h", neigh_feature)])
    neigh_feature = gw.recv(msg, max_recv)
    neigh_feature = fluid.layers.fc(neigh_feature,
                                    hidden_size,
                                    act=act,
                                    name=name + '_r')

    self_feature = feature
    self_feature = fluid.layers.fc(self_feature,
                                   hidden_size,
                                   act=act,
                                   name=name + '_l')
    output = fluid.layers.concat([self_feature, neigh_feature], axis=1)
    output = fluid.layers.l2_normalize(output, axis=1)
    return output


def graphsage_lstm(gw, feature, hidden_size, act, name):
    inner_hidden_size = 128
    neigh_feature = fluid.layers.fc(feature, inner_hidden_size, act="relu")

    hidden_dim = 128
    forward_proj = fluid.layers.fc(input=neigh_feature,
                                   size=hidden_dim * 4,
                                   bias_attr=False,
                                   name="lstm_proj")
    msg = gw.send(copy_send, nfeat_list=[("h", forward_proj)])
    neigh_feature = gw.recv(msg, lstm_recv)
    neigh_feature = fluid.layers.fc(neigh_feature,
                                    hidden_size,
                                    act=act,
                                    name=name + '_r')

    self_feature = feature
    self_feature = fluid.layers.fc(self_feature,
                                   hidden_size,
                                   act=act,
                                   name=name + '_l')
    output = fluid.layers.concat([self_feature, neigh_feature], axis=1)
    output = fluid.layers.l2_normalize(output, axis=1)
    return output


def build_graph_model(graph_wrapper, num_class, k_hop, graphsage_type,
                      hidden_size):
    node_index = fluid.layers.data(
        "node_index", shape=[None], dtype="int64", append_batch_size=False)

    node_label = fluid.layers.data(
        "node_label", shape=[None, 1], dtype="int64", append_batch_size=False)

    #feature = fluid.layers.gather(feature, graph_wrapper.node_feat['feats'])
    feature = graph_wrapper.node_feat['feats']
    feature.stop_gradient = True

    for i in range(k_hop):
        if graphsage_type == 'graphsage_mean':
            feature = graphsage_mean(
                graph_wrapper,
                feature,
                hidden_size,
                act="relu",
                name="graphsage_mean_%s" % i)
        elif graphsage_type == 'graphsage_meanpool':
            feature = graphsage_meanpool(
                graph_wrapper,
                feature,
                hidden_size,
                act="relu",
                name="graphsage_meanpool_%s" % i)
        elif graphsage_type == 'graphsage_maxpool':
            feature = graphsage_maxpool(
                graph_wrapper,
                feature,
                hidden_size,
                act="relu",
                name="graphsage_maxpool_%s" % i)
        elif graphsage_type == 'graphsage_lstm':
            feature = graphsage_lstm(
                graph_wrapper,
                feature,
                hidden_size,
                act="relu",
                name="graphsage_maxpool_%s" % i)
        else:
            raise ValueError("graphsage type %s is not"
                             " implemented" % graphsage_type)

    feature = fluid.layers.gather(feature, node_index)
    logits = fluid.layers.fc(feature,
                             num_class,
                             act=None,
                             name='classification_layer')
    proba = fluid.layers.softmax(logits)

    loss = fluid.layers.softmax_with_cross_entropy(
        logits=logits, label=node_label)
    loss = fluid.layers.mean(loss)
    acc = fluid.layers.accuracy(input=proba, label=node_label, k=1)
    return loss, acc


class GraphsageModel(object):
    def __init__(self, args):
        self.args = args

    def forward(self):
        args = self.args

        graph_wrapper = pgl.graph_wrapper.GraphWrapper(
            "sub_graph", node_feat=[('feats', [None, 602], np.dtype('float32'))])
        loss, acc = build_graph_model(
            graph_wrapper,
            num_class=args.num_class,
            hidden_size=args.hidden_size,
            graphsage_type=args.graphsage_type,
            k_hop=len(args.samples))

        loss.persistable = True

        self.graph_wrapper = graph_wrapper
        self.loss = loss
        self.acc = acc
        return loss

