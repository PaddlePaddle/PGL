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
"""This package implements some common message passing 
functions to help building graph neural networks.
"""

import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as L
from pgl.utils import paddle_helper

__all__ = ['copy_send', 'weighted_copy_send', 'mean_recv', 
        'sum_recv', 'max_recv', 'lstm_recv', 'graphsage_sum',
        'graphsage_mean', 'pinsage_mean', 'pinsage_sum', 
        'softmax_agg', 'msg_norm']


def copy_send(src_feat, dst_feat, edge_feat):
    """doc"""
    return src_feat["h"]

def weighted_copy_send(src_feat, dst_feat, edge_feat):
    """doc"""
    return src_feat["h"] * edge_feat["weight"]

def mean_recv(feat):
    """doc"""
    return L.sequence_pool(feat, pool_type="average")


def sum_recv(feat):
    """doc"""
    return L.sequence_pool(feat, pool_type="sum")


def max_recv(feat):
    """doc"""
    return L.sequence_pool(feat, pool_type="max")


def lstm_recv(hidden_dim):
    """doc"""
    def lstm_recv_inside(feat):
        forward, _ = L.dynamic_lstm(
            input=feat, size=hidden_dim * 4, use_peepholes=False)
        output = L.sequence_last_step(forward)
        return output
    return lstm_recv_inside


def graphsage_sum(gw, feature, hidden_size, act, initializer, learning_rate, name):
    """doc"""
    msg = gw.send(copy_send, nfeat_list=[("h", feature)])
    neigh_feature = gw.recv(msg, sum_recv)
    self_feature = feature
    self_feature = L.fc(self_feature,
                                   hidden_size,
                                   act=act,
                                   param_attr=fluid.ParamAttr(name=name + "_l.w_0", initializer=initializer,
                                   learning_rate=learning_rate),
                                    bias_attr=name+"_l.b_0"
                                   )
    neigh_feature = L.fc(neigh_feature,
                                    hidden_size,
                                    act=act,
                                    param_attr=fluid.ParamAttr(name=name + "_r.w_0", initializer=initializer,
                                   learning_rate=learning_rate),
                                    bias_attr=name+"_r.b_0"
                                    )
    output = L.concat([self_feature, neigh_feature], axis=1)
    output = L.l2_normalize(output, axis=1)
    return output


def graphsage_mean(gw, feature, hidden_size, act, initializer, learning_rate, name):
    """doc"""
    msg = gw.send(copy_send, nfeat_list=[("h", feature)])
    neigh_feature = gw.recv(msg, mean_recv)
    self_feature = feature
    self_feature = L.fc(self_feature,
                                   hidden_size,
                                   act=act,
                                   param_attr=fluid.ParamAttr(name=name + "_l.w_0", initializer=initializer,
                                   learning_rate=learning_rate),
                                    bias_attr=name+"_l.b_0"
                                   )
    neigh_feature = L.fc(neigh_feature,
                                    hidden_size,
                                    act=act,
                                    param_attr=fluid.ParamAttr(name=name + "_r.w_0", initializer=initializer,
                                   learning_rate=learning_rate),
                                    bias_attr=name+"_r.b_0"
                                    )
    output = L.concat([self_feature, neigh_feature], axis=1)
    output = L.l2_normalize(output, axis=1)
    return output


def pinsage_mean(gw, feature, hidden_size, act, initializer, learning_rate, name):
    """doc"""
    msg = gw.send(weighted_copy_send, nfeat_list=[("h", feature)], efeat_list=["weight"])
    neigh_feature = gw.recv(msg, mean_recv)
    self_feature = feature
    self_feature = L.fc(self_feature,
                                   hidden_size,
                                   act=act,
                                   param_attr=fluid.ParamAttr(name=name + "_l.w_0", initializer=initializer,
                                   learning_rate=learning_rate),
                                    bias_attr=name+"_l.b_0"
                                   )
    neigh_feature = L.fc(neigh_feature,
                                    hidden_size,
                                    act=act,
                                    param_attr=fluid.ParamAttr(name=name + "_r.w_0", initializer=initializer,
                                   learning_rate=learning_rate),
                                    bias_attr=name+"_r.b_0"
                                    )
    output = L.concat([self_feature, neigh_feature], axis=1)
    output = L.l2_normalize(output, axis=1)
    return output


def pinsage_sum(gw, feature, hidden_size, act, initializer, learning_rate, name):
    """doc"""
    msg = gw.send(weighted_copy_send, nfeat_list=[("h", feature)], efeat_list=["weight"])
    neigh_feature = gw.recv(msg, sum_recv)
    self_feature = feature
    self_feature = L.fc(self_feature,
                                   hidden_size,
                                   act=act,
                                   param_attr=fluid.ParamAttr(name=name + "_l.w_0", initializer=initializer,
                                   learning_rate=learning_rate),
                                    bias_attr=name+"_l.b_0"
                                   )
    neigh_feature = L.fc(neigh_feature,
                                    hidden_size,
                                    act=act,
                                    param_attr=fluid.ParamAttr(name=name + "_r.w_0", initializer=initializer,
                                   learning_rate=learning_rate),
                                    bias_attr=name+"_r.b_0"
                                    )
    output = L.concat([self_feature, neigh_feature], axis=1)
    output = L.l2_normalize(output, axis=1)
    return output
    
    
def softmax_agg(beta):
    """Implementation of softmax_agg aggregator, see more information in the paper
    "DeeperGCN: All You Need to Train Deeper GCNs"
    (https://arxiv.org/pdf/2006.07739.pdf)

    Args:
        msg: the received message, lod-tensor, (batch_size, seq_len, hidden_size)
        beta: Inverse Temperature

    Return:
        An output tensor with shape (num_nodes, hidden_size)
    """
    
    def softmax_agg_inside(msg):
        alpha = paddle_helper.sequence_softmax(msg, beta)
        msg = msg * alpha
        return L.sequence_pool(msg, "sum")
    
    return softmax_agg_inside


def msg_norm(x, msg, name):
    """Implementation of message normalization, see more information in the paper
    "DeeperGCN: All You Need to Train Deeper GCNs"
    (https://arxiv.org/pdf/2006.07739.pdf)

    Args:
        x: centre node feature (num_nodes, feature_size)
        msg: neighbor node feature (num_nodes, feature_size)
        name: name for s

    Return:
        An output tensor with shape (num_nodes, feature_size)
    """
    s = L.create_parameter(
            shape=[1],
            dtype='float32',
            default_initializer=
                fluid.initializer.ConstantInitializer(value=1.0),
            name=name + '_s_msg_norm')

    msg = L.l2_normalize(msg, axis=1)
    x_norm = L.reduce_sum(x * x, dim=1, keep_dim=True)
    x_norm = L.sqrt(x_norm)
    msg = msg * x_norm * s
    return msg

