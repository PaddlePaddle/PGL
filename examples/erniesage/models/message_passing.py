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
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as L


def copy_send(src_feat, dst_feat, edge_feat):
    """doc"""
    return src_feat["h"]

def weighted_copy_send(src_feat, dst_feat, edge_feat):
    """doc"""
    return src_feat["h"] * edge_feat["weight"]

def mean_recv(feat):
    """doc"""
    return fluid.layers.sequence_pool(feat, pool_type="average")


def sum_recv(feat):
    """doc"""
    return fluid.layers.sequence_pool(feat, pool_type="sum")


def max_recv(feat):
    """doc"""
    return fluid.layers.sequence_pool(feat, pool_type="max")


def lstm_recv(feat):
    """doc"""
    hidden_dim = 128
    forward, _ = fluid.layers.dynamic_lstm(
        input=feat, size=hidden_dim * 4, use_peepholes=False)
    output = fluid.layers.sequence_last_step(forward)
    return output


def graphsage_sum(gw, feature, hidden_size, act, initializer, learning_rate, name):
    """doc"""
    msg = gw.send(copy_send, nfeat_list=[("h", feature)])
    neigh_feature = gw.recv(msg, sum_recv)
    self_feature = feature
    self_feature = fluid.layers.fc(self_feature,
                                   hidden_size,
                                   act=act,
                                   param_attr=fluid.ParamAttr(name=name + "_l", initializer=initializer,
                                   learning_rate=learning_rate),
                                   )
    neigh_feature = fluid.layers.fc(neigh_feature,
                                    hidden_size,
                                    act=act,
                                    param_attr=fluid.ParamAttr(name=name + "_r", initializer=initializer,
                                   learning_rate=learning_rate),
                                    )
    output = fluid.layers.concat([self_feature, neigh_feature], axis=1)
    output = fluid.layers.l2_normalize(output, axis=1)
    return output


def graphsage_mean(gw, feature, hidden_size, act, initializer, learning_rate, name):
    """doc"""
    msg = gw.send(copy_send, nfeat_list=[("h", feature)])
    neigh_feature = gw.recv(msg, mean_recv)
    self_feature = feature
    self_feature = fluid.layers.fc(self_feature,
                                   hidden_size,
                                   act=act,
                                   param_attr=fluid.ParamAttr(name=name + "_l", initializer=initializer,
                                   learning_rate=learning_rate),
                                   )
    neigh_feature = fluid.layers.fc(neigh_feature,
                                    hidden_size,
                                    act=act,
                                    param_attr=fluid.ParamAttr(name=name + "_r", initializer=initializer,
                                   learning_rate=learning_rate),
                                    )
    output = fluid.layers.concat([self_feature, neigh_feature], axis=1)
    output = fluid.layers.l2_normalize(output, axis=1)
    return output


def pinsage_mean(gw, feature, hidden_size, act, initializer, learning_rate, name):
    """doc"""
    msg = gw.send(weighted_copy_send, nfeat_list=[("h", feature)], efeat_list=["weight"])
    neigh_feature = gw.recv(msg, mean_recv)
    self_feature = feature
    self_feature = fluid.layers.fc(self_feature,
                                   hidden_size,
                                   act=act,
                                   param_attr=fluid.ParamAttr(name=name + "_l", initializer=initializer,
                                   learning_rate=learning_rate),
                                   )
    neigh_feature = fluid.layers.fc(neigh_feature,
                                    hidden_size,
                                    act=act,
                                    param_attr=fluid.ParamAttr(name=name + "_r", initializer=initializer,
                                   learning_rate=learning_rate),
                                    )
    output = fluid.layers.concat([self_feature, neigh_feature], axis=1)
    output = fluid.layers.l2_normalize(output, axis=1)
    return output


def pinsage_sum(gw, feature, hidden_size, act, initializer, learning_rate, name):
    """doc"""
    msg = gw.send(weighted_copy_send, nfeat_list=[("h", feature)], efeat_list=["weight"])
    neigh_feature = gw.recv(msg, sum_recv)
    self_feature = feature
    self_feature = fluid.layers.fc(self_feature,
                                   hidden_size,
                                   act=act,
                                   param_attr=fluid.ParamAttr(name=name + "_l", initializer=initializer,
                                   learning_rate=learning_rate),
                                   )
    neigh_feature = fluid.layers.fc(neigh_feature,
                                    hidden_size,
                                    act=act,
                                    param_attr=fluid.ParamAttr(name=name + "_r", initializer=initializer,
                                   learning_rate=learning_rate),
                                    )
    output = fluid.layers.concat([self_feature, neigh_feature], axis=1)
    output = fluid.layers.l2_normalize(output, axis=1)
    return output
