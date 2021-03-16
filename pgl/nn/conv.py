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
"""This package implements common layers to help building
graph neural networks.
"""
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import pgl
from pgl.nn import functional as GF

__all__ = [
    'GCNConv',
    "GATConv",
    'APPNP',
    'GCNII',
    'TransformerConv',
    'GINConv',
    "GraphSageConv",
    "PinSageConv",
]


class GraphSageConv(nn.Layer):
    """ GraphSAGE is a general inductive framework that leverages node feature
    information (e.g., text attributes) to efficiently generate node embeddings
    for previously unseen data.

    Paper reference:
    Hamilton, Will, Zhitao Ying, and Jure Leskovec.
    "Inductive representation learning on large graphs."
    Advances in neural information processing systems. 2017.

    Args:

        input_size: The size of the inputs.

        hidden_size: The size of outputs

        aggr_func: (default "sum") Aggregation function for GraphSage ["sum", "mean", "max", "min"].
    """

    def __init__(self, input_size, hidden_size, aggr_func="sum"):
        super(GraphSageConv, self).__init__()
        assert aggr_func in ["sum", "mean", "max", "min"], \
                "Only support 'sum', 'mean', 'max', 'min' built-in receive function."
        self.aggr_func = "reduce_%s" % aggr_func

        self.self_linear = nn.Linear(input_size, hidden_size)
        self.neigh_linear = nn.Linear(input_size, hidden_size)

    def forward(self, graph, feature, act=None):
        """

        Args:

            graph: `pgl.Graph` instance.

            feature: A tensor with shape (num_nodes, input_size)

            act: (default None) Activation for outputs and before normalize.


        Return:

            A tensor with shape (num_nodes, output_size)

        """

        def _send_func(src_feat, dst_feat, edge_feat):
            return {"msg": src_feat["h"]}

        def _recv_func(message):
            return getattr(message, self.aggr_func)(message["msg"])

        msg = graph.send(_send_func, src_feat={"h": feature})
        neigh_feature = graph.recv(reduce_func=_recv_func, msg=msg)

        self_feature = self.self_linear(feature)
        neigh_feature = self.neigh_linear(neigh_feature)
        output = self_feature + neigh_feature
        if act is not None:
            output = getattr(F, act)(output)

        output = F.normalize(output, axis=1)
        return output


class PinSageConv(nn.Layer):
    """ PinSage combines efficient random walks and graph convolutions to
    generate embeddings of nodes (i.e., items) that incorporate both graph
    structure as well as node feature information.

    Paper reference:
    Ying, Rex, et al.
    "Graph convolutional neural networks for web-scale recommender systems."
    Proceedings of the 24th ACM SIGKDD International Conference on Knowledge
    Discovery & Data Mining. 2018.

    Args:

        input_size: The size of the inputs.

        hidden_size: The size of outputs

        aggr_func: (default "sum") Aggregation function for GraphSage ["sum", "mean", "max", "min"].

    """

    def __init__(self, input_size, hidden_size, aggr_func="sum"):
        super(PinSageConv, self).__init__()
        assert aggr_func in ["sum", "mean", "max", "min"], \
                "Only support 'sum', 'mean', 'max', 'min' built-in receive function."
        self.aggr_func = "reduce_%s" % aggr_func

        self.self_linear = nn.Linear(input_size, hidden_size)
        self.neigh_linear = nn.Linear(input_size, hidden_size)

    def forward(self, graph, nfeat, efeat, act=None):
        """
        Args:

            graph: `pgl.Graph` instance.

            nfeat: A tensor with shape (num_nodes, input_size)

            efeat: A tensor with shape (num_edges, 1) denotes edge weight.

            act: (default None) Activation for outputs and before normalize.

        Return:

            A tensor with shape (num_nodes, output_size)
        """

        def _send_func(src_feat, dst_feat, edge_feat):
            return {'msg': src_feat["h"] * edge_feat["w"]}

        def _recv_func(message):
            return getattr(message, self.aggr_func)(message["msg"])

        msg = graph.send(
            _send_func, src_feat={"h": nfeat}, edge_feat={"w": efeat})
        neigh_feature = graph.recv(reduce_func=_recv_func, msg=msg)

        self_feature = self.self_linear(nfeat)
        neigh_feature = self.neigh_linear(neigh_feature)
        output = self_feature + neigh_feature
        if act is not None:
            output = getattr(F, act)(output)

        output = F.normalize(output, axis=1)
        return output


class GCNConv(nn.Layer):
    """Implementation of graph convolutional neural networks (GCN)

    This is an implementation of the paper SEMI-SUPERVISED CLASSIFICATION
    WITH GRAPH CONVOLUTIONAL NETWORKS (https://arxiv.org/pdf/1609.02907.pdf).

    Args:

        input_size: The size of the inputs.

        output_size: The size of outputs

        activation: The activation for the output.

        norm: If :code:`norm` is True, then the feature will be normalized.

    """

    def __init__(self, input_size, output_size, activation=None, norm=True):
        super(GCNConv, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = nn.Linear(input_size, output_size, bias_attr=False)
        self.bias = self.create_parameter(shape=[output_size], is_bias=True)
        self.norm = norm
        if isinstance(activation, str):
            activation = getattr(F, activation)
        self.activation = activation

    def forward(self, graph, feature, norm=None):
        """

        Args:

            graph: `pgl.Graph` instance.

            feature: A tensor with shape (num_nodes, input_size)

            norm: (default None). If :code:`norm` is not None, then the feature will be normalized by given norm. If :code:`norm` is None and :code:`self.norm` is `true`, then we use `lapacian degree norm`.

        Return:

            A tensor with shape (num_nodes, output_size)

        """

        if self.norm and norm is None:
            norm = GF.degree_norm(graph)

        if self.input_size > self.output_size:
            feature = self.linear(feature)

        if norm is not None:
            feature = feature * norm

        output = graph.send_recv(feature, "sum")

        if self.input_size <= self.output_size:
            output = self.linear(output)

        if norm is not None:
            output = output * norm
        output = output + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output


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

        self.linear = nn.Linear(input_size, num_heads * hidden_size)
        self.weight_src = self.create_parameter(shape=[num_heads, hidden_size])
        self.weight_dst = self.create_parameter(shape=[num_heads, hidden_size])

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


class APPNP(nn.Layer):
    """Implementation of APPNP of "Predict then Propagate: Graph Neural Networks
    meet Personalized PageRank"  (ICLR 2019).

    Args:

        k_hop: K Steps for Propagation

        alpha: The hyperparameter of alpha in the paper.

    Return:
        A tensor with shape (num_nodes, hidden_size)
    """

    def __init__(self, alpha=0.2, k_hop=10):
        super(APPNP, self).__init__()
        self.alpha = alpha
        self.k_hop = k_hop

    def forward(self, graph, feature, norm=None):
        """

        Args:

            graph: `pgl.Graph` instance.

            feature: A tensor with shape (num_nodes, input_size)

            norm: (default None). If :code:`norm` is not None, then the feature will be normalized by given norm. If :code:`norm` is None, then we use `lapacian degree norm`.

        Return:

            A tensor with shape (num_nodes, output_size)

        """
        if norm is None:
            norm = GF.degree_norm(graph)
        h0 = feature

        for _ in range(self.k_hop):
            feature = feature * norm
            feature = graph.send_recv(feature)
            feature = feature * norm
            feature = self.alpha * h0 + (1 - self.alpha) * feature

        return feature


class GCNII(nn.Layer):
    """Implementation of GCNII of "Simple and Deep Graph Convolutional Networks"

    paper: https://arxiv.org/pdf/2007.02133.pdf

    Args:
        hidden_size: The size of inputs and outputs.

        activation: The activation for the output.

        k_hop: Number of layers for gcnii.

        lambda_l: The hyperparameter of lambda in the paper.

        alpha: The hyperparameter of alpha in the paper.

        dropout: Feature dropout rate.
    """

    def __init__(self,
                 hidden_size,
                 activation=None,
                 lambda_l=0.5,
                 alpha=0.2,
                 k_hop=10,
                 dropout=0.6):
        super(GCNII, self).__init__()
        self.hidden_size = hidden_size
        self.activation = activation
        self.lambda_l = lambda_l
        self.alpha = alpha
        self.k_hop = k_hop
        self.dropout = dropout
        self.drop_fn = nn.Dropout(dropout)
        self.mlps = nn.LayerList()
        for _ in range(k_hop):
            self.mlps.append(nn.Linear(hidden_size, hidden_size))
        if isinstance(activation, str):
            activation = getattr(F, activation)
        self.activation = activation

    def forward(self, graph, feature, norm=None):
        """
        Args:

            graph: `pgl.Graph` instance.

            feature: A tensor with shape (num_nodes, input_size)

            norm: (default None). If :code:`norm` is not None, then the feature will be normalized by given norm. If :code:`norm` is None, then we use `lapacian degree norm`.

        Return:

            A tensor with shape (num_nodes, output_size)

        """

        if norm is None:
            norm = GF.degree_norm(graph)
        h0 = feature

        for i in range(self.k_hop):
            beta_i = np.log(1.0 * self.lambda_l / (i + 1) + 1)
            feature = self.drop_fn(feature)

            feature = feature * norm
            feature = graph.send_recv(feature)
            feature = feature * norm
            feature = self.alpha * h0 + (1 - self.alpha) * feature

            feature_transed = self.mlps[i](feature)
            feature = beta_i * feature_transed + (1 - beta_i) * feature
            if self.activation is not None:
                feature = self.activation(feature)
        return feature


class TransformerConv(nn.Layer):
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

        self.q = nn.Linear(input_size, num_heads * hidden_size)
        self.k = nn.Linear(input_size, num_heads * hidden_size)
        self.v = nn.Linear(input_size, num_heads * hidden_size)

        self.feat_dropout = nn.Dropout(p=feat_drop)
        self.attn_dropout = nn.Dropout(p=attn_drop)

        if skip_feat:
            if concat:
                self.skip_feat = nn.Linear(input_size, num_heads * hidden_size)
            else:
                self.skip_feat = nn.Linear(input_size, hidden_size)
        else:
            self.skip_feat = None

        if gate:
            if concat:
                self.gate = nn.Linear(3 * num_heads * hidden_size, 1)
            else:
                self.gate = nn.Linear(3 * hidden_size, 1)
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


class GINConv(nn.Layer):
    """Implementation of Graph Isomorphism Network (GIN) layer.

    This is an implementation of the paper How Powerful are Graph Neural Networks?
    (https://arxiv.org/pdf/1810.00826.pdf).
    In their implementation, all MLPs have 2 layers. Batch normalization is applied
    on every hidden layer.

    Args:

        input_size: The size of input.

        output_size: The size of output.

        activation: The activation for the output.

        init_eps: float, optional
            Initial :math:`\epsilon` value, default is 0.

        train_eps: bool, optional
            if True, :math:`\epsilon` will be a learnable parameter.

    """

    def __init__(self,
                 input_size,
                 output_size,
                 activation=None,
                 init_eps=0.0,
                 train_eps=False):
        super(GINConv, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear1 = nn.Linear(input_size, output_size, bias_attr=True)
        self.linear2 = nn.Linear(output_size, output_size, bias_attr=True)
        self.layer_norm = nn.LayerNorm(output_size)
        if train_eps:
            self.epsilon = self.create_parameter(
                shape=[1, 1],
                dtype='float32',
                default_initializer=nn.initializer.Constant(value=init_eps))
        else:
            self.epsilon = init_eps

        if isinstance(activation, str):
            activation = getattr(F, activation)
        self.activation = activation

    def forward(self, graph, feature):
        """

        Args:

            graph: `pgl.Graph` instance.

            feature: A tensor with shape (num_nodes, input_size)

        Return:

            A tensor with shape (num_nodes, output_size)
        """
        neigh_feature = graph.send_recv(feature, reduce_func="sum")
        output = neigh_feature + feature * (self.epsilon + 1.0)

        output = self.linear1(output)
        output = self.layer_norm(output)

        if self.activation is not None:
            output = self.activation(output)

        output = self.linear2(output)

        return output
