# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import math
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import pgl
from pgl.nn import functional as GF


def linear_init(input_size, hidden_size, with_bias=True, init_type='gcn'):
    if init_type == 'gcn':
        fc_w_attr = paddle.ParamAttr(initializer=nn.initializer.XavierNormal())
        fc_bias_attr = paddle.ParamAttr(
            initializer=nn.initializer.Constant(0.0))
    else:
        fan_in = input_size
        bias_bound = 1.0 / math.sqrt(fan_in)
        fc_bias_attr = paddle.ParamAttr(initializer=nn.initializer.Uniform(
            low=-bias_bound, high=bias_bound))

        negative_slope = math.sqrt(5)
        gain = math.sqrt(2.0 / (1 + negative_slope**2))
        std = gain / math.sqrt(fan_in)
        weight_bound = math.sqrt(3.0) * std
        fc_w_attr = paddle.ParamAttr(initializer=nn.initializer.Uniform(
            low=-weight_bound, high=weight_bound))

    if not with_bias:
        fc_bias_attr = False

    return nn.Linear(
        input_size, hidden_size, weight_attr=fc_w_attr, bias_attr=fc_bias_attr)


class GATConv(nn.Layer):
    """Implementation of graph attention networks (GAT)
    This is an implementation of the paper GRAPH ATTENTION NETWORKS
    (https://arxiv.org/abs/1710.10903).
    Args:
        input_size: The size of the inputs. 
        hidden_size: The hidden size for gat.
        activation: (default None) The activation for the output.
        num_heads: (default 1) The head number in gat.
        feat_drop: (default 0.6) Dropout rate for feature.
        attn_drop: (default 0.6) Dropout rate for attention.
        concat: (default True) Whether to concat output heads or average them.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 feat_drop=0.6,
                 attn_drop=0.6,
                 num_heads=1,
                 concat=True,
                 activation=None):
        super(GATConv, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.concat = concat

        self.linear = linear_init(
            input_size, num_heads * hidden_size, init_type='gcn')

        fc_w_attr = paddle.ParamAttr(initializer=nn.initializer.XavierNormal())
        self.weight_src = self.create_parameter(
            shape=[1, num_heads, hidden_size], attr=fc_w_attr)

        fc_w_attr = paddle.ParamAttr(initializer=nn.initializer.XavierNormal())
        self.weight_dst = self.create_parameter(
            shape=[1, num_heads, hidden_size], attr=fc_w_attr)

        self.feat_dropout = nn.Dropout(p=feat_drop)
        self.attn_dropout = nn.Dropout(p=attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        if isinstance(activation, str):
            activation = getattr(F, activation)
        self.activation = activation

    def _send_attention(self, src_feat, dst_feat, edge_feat):
        alpha = src_feat["src"] + dst_feat["dst"]
        alpha = self.leaky_relu(alpha)
        return {"alpha": alpha, "h": src_feat["h"]}

    def _reduce_attention(self, msg):
        alpha = msg.reduce_softmax(msg["alpha"])
        alpha = paddle.reshape(alpha, [-1, self.num_heads, 1])
        if self.attn_drop > 1e-15:
            alpha = self.attn_dropout(alpha)

        feature = msg["h"]
        feature = paddle.reshape(feature,
                                 [-1, self.num_heads, self.hidden_size])
        feature = feature * alpha
        if self.concat:
            feature = paddle.reshape(feature,
                                     [-1, self.num_heads * self.hidden_size])
        else:
            feature = paddle.mean(feature, axis=1)

        feature = msg.reduce(feature, pool_type="sum")
        return feature

    def forward(self, graph, feature):
        """
         
        Args:
 
            graph: `pgl.Graph` instance.
            feature: A tensor with shape (num_nodes, input_size)
     
        Return:
            If `concat=True` then return a tensor with shape (num_nodes, hidden_size),
            else return a tensor with shape (num_nodes, hidden_size * num_heads) 
        """

        if self.feat_drop > 1e-15:
            feature = self.feat_dropout(feature)

        feature = self.linear(feature)
        feature = paddle.reshape(feature,
                                 [-1, self.num_heads, self.hidden_size])

        attn_src = paddle.sum(feature * self.weight_src, axis=-1)
        attn_dst = paddle.sum(feature * self.weight_dst, axis=-1)
        msg = graph.send(
            self._send_attention,
            src_feat={"src": attn_src,
                      "h": feature},
            dst_feat={"dst": attn_dst})
        output = graph.recv(reduce_func=self._reduce_attention, msg=msg)

        if self.activation is not None:
            output = self.activation(output)
        return output


class TransformerConv(nn.Layer):
    """Implementation of TransformerConv from UniMP
    This is an implementation of the paper Unified Message Passing Model for Semi-Supervised Classification
    (https://arxiv.org/abs/2009.03509).
    Args:
    
        input_size: The size of the inputs. 
 
        hidden_size: The hidden size for gat.
 
        activation: (default None) The activation for the output.
 
        num_heads: (default 4) The head number in transformerconv.
 
        feat_drop: (default 0.6) Dropout rate for feature.
 
        attn_drop: (default 0.6) Dropout rate for attention.
 
        concat: (default True) Whether to concat output heads or average them.
        skip_feat: (default True) Whether to add a skip conect from input to output.
        gate: (default False) Whether to use a gate function in skip conect.
        layer_norm: (default True) Whether to aply layer norm in output
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_heads=4,
                 feat_drop=0.6,
                 attn_drop=0.6,
                 concat=True,
                 skip_feat=True,
                 gate=False,
                 layer_norm=True,
                 activation='relu'):
        super(TransformerConv, self).__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.concat = concat

        self.q = linear_init(
            input_size, num_heads * hidden_size, init_type='gcn')
        self.k = linear_init(
            input_size, num_heads * hidden_size, init_type='gcn')
        self.v = linear_init(
            input_size, num_heads * hidden_size, init_type='gcn')

        self.feat_dropout = nn.Dropout(p=feat_drop)
        self.attn_dropout = nn.Dropout(p=attn_drop)

        if skip_feat:
            if concat:
                self.skip_feat = linear_init(
                    input_size, num_heads * hidden_size, init_type='gcn')
            else:
                self.skip_feat = linear_init(
                    input_size, hidden_size, init_type='gcn')
        else:
            self.skip_feat = None

        if gate:
            if concat:
                self.gate = linear_init(
                    3 * num_heads * hidden_size, 1, init_type='gcn')
            else:
                self.gate = linear_init(3 * hidden_size, 1, init_type='gcn')
        else:
            self.gate = None

        if layer_norm:
            if self.concat:
                self.layer_norm = nn.LayerNorm(num_heads * hidden_size)
            else:
                self.layer_norm = nn.LayerNorm(hidden_size)
        else:
            self.layer_norm = None

        if isinstance(activation, str):
            activation = getattr(F, activation)
        self.activation = activation

    def send_attention(self, src_feat, dst_feat, edge_feat):
        if "edge_feat" in edge_feat:
            alpha = dst_feat["q"] * (src_feat["k"] + edge_feat['edge_feat'])
            src_feat["v"] = src_feat["v"] + edge_feat["edge_feat"]
        else:
            alpha = dst_feat["q"] * src_feat["k"]
        alpha = paddle.sum(alpha, axis=-1)
        return {"alpha": alpha, "v": src_feat["v"]}

    def reduce_attention(self, msg):
        alpha = msg.reduce_softmax(msg["alpha"])
        alpha = paddle.reshape(alpha, [-1, self.num_heads, 1])
        if self.attn_drop > 1e-15:
            alpha = self.attn_dropout(alpha)

        feature = msg["v"]
        feature = feature * alpha
        if self.concat:
            feature = paddle.reshape(feature,
                                     [-1, self.num_heads * self.hidden_size])
        else:
            feature = paddle.mean(feature, axis=1)
        feature = msg.reduce(feature, pool_type="sum")
        return feature

    def send_recv(self, graph, q, k, v, edge_feat):
        q = q / (self.hidden_size**0.5)
        if edge_feat is not None:
            msg = graph.send(
                self.send_attention,
                src_feat={'k': k,
                          'v': v},
                dst_feat={'q': q},
                edge_feat={'edge_feat': edge_feat})
        else:
            msg = graph.send(
                self.send_attention,
                src_feat={'k': k,
                          'v': v},
                dst_feat={'q': q})

        output = graph.recv(reduce_func=self.reduce_attention, msg=msg)
        return output

    def forward(self, graph, feature, edge_feat=None):
        if self.feat_drop > 1e-5:
            feature = self.feat_dropout(feature)
        q = self.q(feature)
        k = self.k(feature)
        v = self.v(feature)

        q = paddle.reshape(q, [-1, self.num_heads, self.hidden_size])
        k = paddle.reshape(k, [-1, self.num_heads, self.hidden_size])
        v = paddle.reshape(v, [-1, self.num_heads, self.hidden_size])
        if edge_feat is not None:
            if self.feat_dropout > 1e-5:
                edge_feat = self.feat_dropout(edge_feat)
            edge_feat = paddle.reshape(edge_feat,
                                       [-1, self.num_heads, self.hidden_size])

        output = self.send_recv(graph, q, k, v, edge_feat=edge_feat)

        if self.skip_feat is not None:
            skip_feat = self.skip_feat(feature)
            if self.gate is not None:
                gate = F.sigmoid(
                    self.gate(
                        paddle.concat(
                            [skip_feat, output, skip_feat - output], axis=-1)))
                output = gate * skip_feat + (1 - gate) * output
            else:
                output = skip_feat + output

        if self.layer_norm is not None:
            output = self.layer_norm(output)

        if self.activation is not None:
            output = self.activation(output)
        return output
