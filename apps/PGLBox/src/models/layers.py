# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved
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
Doc String
"""
import paddle
import paddle.fluid as fluid
import pgl


def gin(graph, feature, next_num_nodes, hidden_size, act, name):
    """doc"""
    src, dst = graph.edges[:, 0], graph.edges[:, 1]
    neigh_feature = paddle.geometric.send_u_recv(
        feature, src, dst, pool_type="sum", out_size=next_num_nodes)
    self_feature = feature[:next_num_nodes]
    output = self_feature + neigh_feature
    output = fluid.layers.fc(output,
                             hidden_size,
                             act=act,
                             param_attr=fluid.ParamAttr(name=name + '_w'),
                             bias_attr=fluid.ParamAttr(name=name + '_b'))
    output = output + feature[:next_num_nodes]
    return output

def graphsage_mean_efeat(graph, feature, edge_feat, next_num_nodes, hidden_size, act, name):
    """ graphsage mean efeat (do not support efeat currently, don't run!!!) """
    src, dst = graph.edges[:, 0], graph.edges[:, 1]
    neigh_feature = paddle.geometric.send_ue_recv(
        feature, edge_feat, src, dst, "mul", "sum", out_size=next_num_nodes)
    self_feature = feature[:next_num_nodes]
    output = fluid.layers.concat([self_feature, neigh_feature], axis=1)
    output = fluid.layers.fc(output,
                            hidden_size,
                            act=act,
                            param_attr=fluid.ParamAttr(name=name + '_w'),
                            bias_attr=fluid.ParamAttr(name=name + '_b'))
    
    output = fluid.layers.l2_normalize(output, axis=1)
    return output

def graphsage_mean(graph, feature, next_num_nodes, hidden_size, act, name):
    """doc"""
    src, dst = graph.edges[:, 0], graph.edges[:, 1]
    neigh_feature = paddle.geometric.send_u_recv(
        feature, src, dst, pool_type="mean", out_size=next_num_nodes)
    self_feature = feature[:next_num_nodes]
    output = fluid.layers.concat([self_feature, neigh_feature], axis=1)
    output = fluid.layers.fc(output,
                             hidden_size,
                             act=act,
                             param_attr=fluid.ParamAttr(name=name + '_w'),
                             bias_attr=fluid.ParamAttr(name=name + '_b'))
    output = fluid.layers.l2_normalize(output, axis=1)
    return output

def graphsage_bow(graph, feature, next_num_nodes, hidden_size, act, name):
    """doc"""
    src, dst = graph.edges[:, 0], graph.edges[:, 1]
    neigh_feature = paddle.geometric.send_u_recv(
        feature, src, dst, pool_type="mean", out_size=next_num_nodes)
    self_feature = feature[:next_num_nodes]
     
    output = self_feature + neigh_feature
    output = fluid.layers.l2_normalize(output, axis=1)
    return output

def graphsage_meanpool(graph,
                       feature,
                       next_num_nodes,
                       hidden_size,
                       act,
                       name,
                       inner_hidden_size=512):
    """doc"""
    src, dst = graph.edges[:, 0], graph.edges[:, 1]
    neigh_feature = fluid.layers.fc(feature, inner_hidden_size, act="relu")
    neigh_feature = paddle.geometric.send_u_recv(
        neigh_feature, src, dst, pool_type="mean", out_size=next_num_nodes)
    neigh_feature = fluid.layers.fc(neigh_feature,
                                    hidden_size,
                                    act=act,
                                    name=name + '_r')
    self_feature = feature[:next_num_nodes]
    self_feature = fluid.layers.fc(self_feature,
                                   hidden_size,
                                   act=act,
                                   name=name + '_l')
    output = fluid.layers.concat([self_feature, neigh_feature], axis=1)
    output = fluid.layers.l2_normalize(output, axis=1)
    return output

def graphsage_maxpool(graph,
                      feature,
                      next_num_nodes,
                      hidden_size,
                      act,
                      name,
                      inner_hidden_size=512):
    """doc"""
    src, dst = graph.edges[:, 0], graph.edges[:, 1]
    neigh_feature = fluid.layers.fc(feature, inner_hidden_size, act="relu")
    neigh_feature = paddle.geometric.send_u_recv(
        neigh_feature, src, dst, pool_type="max", out_size=next_num_nodes) 
    neigh_feature = fluid.layers.fc(neigh_feature,
                                    hidden_size,
                                    act=act,
                                    name=name + '_r')
    self_feature = feature[:next_num_nodes]
    self_feature = fluid.layers.fc(self_feature,
                                   hidden_size,
                                   act=act,
                                   name=name + '_l')
    output = fluid.layers.concat([self_feature, neigh_feature], axis=1)
    output = fluid.layers.l2_normalize(output, axis=1)
    return output

def gat(graph, feature, next_num_nodes, hidden_size, act, name):
    """ gat """
    neigh_feature = pgl.nn.GATConv(
        input_size=feature.shape[1],
        hidden_size=hidden_size,
        num_heads=1,
        feat_drop=0,
        attn_drop=0,
        activation="relu")(graph, feature)[:next_num_nodes]
    self_feature = feature[:next_num_nodes]
    output = fluid.layers.concat([self_feature, neigh_feature], axis=1)
    output = fluid.layers.fc(output,
                            hidden_size,
                            act=act,
                            param_attr=fluid.ParamAttr(name=name + '_self_w'),
                            bias_attr=fluid.ParamAttr(name=name + '_self_b'))
    return output

def lightgcn(graph, feature, next_num_nodes, hidden_size, act, name):
    """doc
    Notes: currently do not support edge weight.
    """
    src, dst = graph.edges[:, 0], graph.edges[:, 1]
    neigh_feature = paddle.geometric.send_u_recv(
        feature, src, dst, pool_type="sum", out_size=next_num_nodes)
    return neigh_feature
