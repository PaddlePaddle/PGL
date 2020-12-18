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
import paddle
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
