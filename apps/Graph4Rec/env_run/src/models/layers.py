#-*- coding: utf-8 -*-
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import pgl
from pgl.nn import functional as GF

__all__ = [
    "GraphSageSumConv",
    "GraphSageMeanConv",
    "GINConv",
    "GATConv",
    "LightGCNConv",
    "GATNEInteract",
]


class GraphSageSumConv(nn.Layer):
    def __init__(self, input_size, output_size, act=None):
        super(GraphSageSumConv, self).__init__()
        self.aggr_func = "reduce_sum"

        self.self_linear = nn.Linear(input_size, output_size)
        self.neigh_linear = nn.Linear(input_size, output_size)

        if isinstance(act, str):
            act = getattr(F, act)
        self.activation = act

    def forward(self, graph, feature):
        def _send_func(src_feat, dst_feat, edge_feat):
            return {"msg": src_feat["h"]}

        def _recv_func(message):
            return getattr(message, self.aggr_func)(message["msg"])

        msg = graph.send(_send_func, src_feat={"h": feature})
        neigh_feature = graph.recv(reduce_func=_recv_func, msg=msg)

        self_feature = self.self_linear(feature)
        neigh_feature = self.neigh_linear(neigh_feature)
        output = self_feature + neigh_feature

        if self.activation is not None:
            output = self.activation(output)

        return output


class GraphSageMeanConv(nn.Layer):
    def __init__(self, input_size, output_size, act=None):
        super(GraphSageMeanConv, self).__init__()
        self.aggr_func = "reduce_mean"

        self.self_linear = nn.Linear(input_size, output_size)
        self.neigh_linear = nn.Linear(input_size, output_size)

        if isinstance(act, str):
            act = getattr(F, act)
        self.activation = act

    def forward(self, graph, feature):
        def _send_func(src_feat, dst_feat, edge_feat):
            return {"msg": src_feat["h"]}

        def _recv_func(message):
            return getattr(message, self.aggr_func)(message["msg"])

        msg = graph.send(_send_func, src_feat={"h": feature})
        neigh_feature = graph.recv(reduce_func=_recv_func, msg=msg)

        self_feature = self.self_linear(feature)
        neigh_feature = self.neigh_linear(neigh_feature)
        output = self_feature + neigh_feature

        if self.activation is not None:
            output = self.activation(output)

        return output


class GINConv(nn.Layer):
    def __init__(self, input_size, output_size, act=None):
        super(GINConv, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.linear1 = nn.Linear(input_size, output_size, bias_attr=True)
        self.linear2 = nn.Linear(output_size, output_size, bias_attr=True)

        self.epsilon = self.create_parameter(
            shape=[1, 1],
            dtype='float32',
            default_initializer=nn.initializer.Constant(value=1.0))

        if isinstance(act, str):
            act = getattr(F, act)
        self.activation = act

    def forward(self, graph, feature):
        neigh_feature = graph.send_recv(feature, reduce_func="sum")
        output = neigh_feature + feature * (self.epsilon + 1.0)
        output = self.linear1(output)

        if self.activation is not None:
            output = self.activation(output)

        output = self.linear2(output)

        return output


class GATConv(nn.Layer):
    def __init__(self, input_size, output_size, act=None):
        super(GATConv, self).__init__()
        self.output_size = output_size
        self.num_heads = 1
        self.feat_drop = 0.6
        self.attn_drop = 0.6
        self.concat = True

        self.linear = nn.Linear(input_size, self.num_heads * output_size)
        self.weight_src = self.create_parameter(
            shape=[self.num_heads, output_size])
        self.weight_dst = self.create_parameter(
            shape=[self.num_heads, output_size])

        self.feat_dropout = nn.Dropout(p=self.feat_drop)
        self.attn_dropout = nn.Dropout(p=self.attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        if isinstance(act, str):
            act = getattr(F, act)
        self.activation = act

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
                                 [-1, self.num_heads, self.output_size])
        feature = feature * alpha
        if self.concat:
            feature = paddle.reshape(feature,
                                     [-1, self.num_heads * self.output_size])
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

            If `concat=True` then return a tensor with shape (num_nodes, output_size),
            else return a tensor with shape (num_nodes, output_size * num_heads) 

        """

        if self.feat_drop > 1e-15:
            feature = self.feat_dropout(feature)

        feature = self.linear(feature)
        feature = paddle.reshape(feature,
                                 [-1, self.num_heads, self.output_size])

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


class LightGCNConv(nn.Layer):
    """
    
    Implementation of LightGCN
    
    This is an implementation of the paper LightGCN: Simplifying 
    and Powering Graph Convolution Network for Recommendation 
    (https://dl.acm.org/doi/10.1145/3397271.3401063).

    """

    def __init__(self, input_size, output_size, act=None):
        super(LightGCNConv, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, graph, feature):
        """
        Args:
 
            graph: `pgl.Graph` instance.

            feature: A tensor with shape (num_nodes, input_size)

        Return:

            A tensor with shape (num_nodes, output_size)

        """
        feature = graph.send_recv(feature, "sum")
        return feature


class GATNEInteract(nn.Layer):
    def __init__(self, hidden_size):
        super(GATNEInteract, self).__init__()
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, feature_list):
        # stack [num_nodes, num_etype, hidden_size]
        U = paddle.stack(feature_list, axis=1)
        # [num_nodes * num_etype, hidden_size]
        tmp = paddle.reshape(U, shape=[-1, self.hidden_size])
        tmp = self.fc1(tmp)
        tmp = paddle.tanh(tmp)
        tmp = self.fc2(tmp)
        #  [num_nodes, num_etype]
        tmp = paddle.reshape(tmp, shape=[-1, len(feature_list)])
        # [num_nodes, 1, num_etype]
        a = paddle.unsqueeze(
            paddle.nn.functional.softmax(
                tmp, axis=-1), axis=1)
        out = paddle.squeeze(paddle.matmul(a, U), axis=1)
        return out
