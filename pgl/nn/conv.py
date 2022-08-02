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
    "GCNConv",
    "GATConv",
    "APPNP",
    "GCNII",
    "TransformerConv",
    "GINConv",
    "GraphSageConv",
    "PinSageConv",
    "RGCNConv",
    "SGCConv",
    "SSGCConv",
    "NGCFConv",
    "LightGCNConv",
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

        self_loop: Whether add self loop in APPNP layer.

    Return:
        A tensor with shape (num_nodes, hidden_size)
    """

    def __init__(self, alpha=0.2, k_hop=10, self_loop=False):
        super(APPNP, self).__init__()
        self.alpha = alpha
        self.k_hop = k_hop
        self.self_loop = self_loop

    def forward(self, graph, feature, norm=None):
        """
         
        Args:
 
            graph: `pgl.Graph` instance.

            feature: A tensor with shape (num_nodes, input_size)

            norm: (default None). If :code:`norm` is not None, then the feature will be normalized by given norm. If :code:`norm` is None, then we use `lapacian degree norm`.
     
        Return:

            A tensor with shape (num_nodes, output_size)

        """
        if self.self_loop:
            index = paddle.arange(start=0, end=graph.num_nodes, dtype="int64")
            self_loop_edges = paddle.transpose(
                paddle.stack((index, index)), [1, 0])

            mask = graph.edges[:, 0] != graph.edges[:, 1]
            mask_index = paddle.masked_select(
                paddle.arange(end=graph.num_edges), mask)
            edges = paddle.gather(graph.edges, mask_index)  # remove self loop

            edges = paddle.concat((self_loop_edges, edges), axis=0)
            graph = pgl.Graph(num_nodes=graph.num_nodes, edges=edges)

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
            if self.feat_drop > 1e-5:
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


class RGCNConv(nn.Layer):
    """Implementation of Relational Graph Convolutional Networks (R-GCN)

    This is an implementation of the paper 
    Modeling Relational Data with Graph Convolutional Networks 
    (http://arxiv.org/abs/1703.06103).

    Args:
        
        in_dim: The input dimension.

        out_dim: The output dimension.

        etypes: A list of edge types of the heterogeneous graph.

        num_bases: int, number of basis decomposition. Details can be found in the paper.

    """

    def __init__(self, in_dim, out_dim, etypes, num_bases=0):
        super(RGCNConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.etypes = etypes
        self.num_rels = len(self.etypes)
        self.num_bases = num_bases

        if self.num_bases <= 0 or self.num_bases >= self.num_rels:
            self.num_bases = self.num_rels

        self.weight = self.create_parameter(
            shape=[self.num_bases, self.in_dim, self.out_dim])

        if self.num_bases < self.num_rels:
            self.w_comp = self.create_parameter(
                shape=[self.num_rels, self.num_bases])

    def forward(self, graph, feat):
        """
        Args:
            graph: `pgl.HeterGraph` instance or a dictionary of `pgl.Graph` with their edge type.

            feat: A tensor with shape (num_nodes, in_dim)
        """

        if self.num_bases < self.num_rels:
            weight = paddle.transpose(self.weight, perm=[1, 0, 2])
            weight = paddle.matmul(self.w_comp, weight)
            # [num_rels, in_dim, out_dim]
            weight = paddle.transpose(weight, perm=[1, 0, 2])
        else:
            weight = self.weight

        def send_func(src_feat, dst_feat, edge_feat):
            return src_feat

        def recv_func(msg):
            return msg.reduce_mean(msg["h"])

        feat_list = []
        for idx, etype in enumerate(self.etypes):
            w = weight[idx, :, :].squeeze()
            h = paddle.matmul(feat, w)
            msg = graph[etype].send(send_func, src_feat={"h": h})
            h = graph[etype].recv(recv_func, msg)
            feat_list.append(h)

        h = paddle.stack(feat_list, axis=0)
        h = paddle.sum(h, axis=0)

        return h


class SGCConv(nn.Layer):
    """Implementation of simplified graph convolutional neural networks (SGC)

    This is an implementation of the paper Simplifying Graph Convolutional Networks
    (https://arxiv.org/pdf/1902.07153.pdf).

    Args:

        input_size: The size of the inputs. 

        output_size: The size of outputs

        k_hop: K Steps for Propagation

        activation: The activation for the output.

        cached: If :code:`cached` is True, then the graph convolution will be pre-computed and stored.

    """

    def __init__(self,
                 input_size,
                 output_size,
                 k_hop=2,
                 cached=True,
                 activation=None,
                 bias=False):
        super(SGCConv, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.k_hop = k_hop
        self.linear = nn.Linear(input_size, output_size, bias_attr=False)
        if bias:
            self.bias = self.create_parameter(
                shape=[output_size], is_bias=True)

        self.cached = cached
        self.cached_output = None
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
        if self.cached:
            if self.cached_output is None:
                norm = GF.degree_norm(graph)
                for hop in range(self.k_hop):
                    feature = feature * norm
                    feature = graph.send_recv(feature, "sum")
                    feature = feature * norm
                self.cached_output = feature
            else:
                feature = self.cached_output
        else:
            norm = GF.degree_norm(graph)
            for hop in range(self.k_hop):
                feature = feature * norm
                feature = graph.send_recv(feature, "sum")
                feature = feature * norm

        output = self.linear(feature)
        if hasattr(self, "bias"):
            output = output + self.bias

        if self.activation is not None:
            output = self.activation(output)
        return output


class SSGCConv(nn.Layer):
    """Implementation of Simple Spectral Graph Convolution (SSGC)

    This is an implementation of the paper Simple Spectral Graph Convolution 
    (https://openreview.net/forum?id=CYO5T-YjWZV).

    Args:

        input_size: The size of the inputs. 

        output_size: The size of outputs

        k_hop: K Steps for Propagation

        alpha: The hyper parameter in paper. 

        activation: The activation for the output.

        cached: If :code:`cached` is True, then the graph convolution will be pre-computed and stored.

    """

    def __init__(self,
                 input_size,
                 output_size,
                 k_hop=16,
                 alpha=0.05,
                 cached=True,
                 activation=None,
                 bias=False):
        super(SSGCConv, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.k_hop = k_hop
        self.alpha = alpha
        self.linear = nn.Linear(input_size, output_size, bias_attr=False)
        if bias:
            self.bias = self.create_parameter(
                shape=[output_size], is_bias=True)

        self.cached = cached
        self.cached_output = None
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
        if self.cached:
            if self.cached_output is None:
                norm = GF.degree_norm(graph)
                ori_feature = feature
                sum_feature = feature
                for hop in range(self.k_hop):
                    feature = feature * norm
                    feature = graph.send_recv(feature, "sum")
                    feature = feature * norm
                    feature = (1 - self.alpha) * feature
                    sum_feature += feature
                feature = sum_feature / self.k_hop + self.alpha * ori_feature
                self.cached_output = feature
            else:
                feature = self.cached_output
        else:

            norm = GF.degree_norm(graph)
            ori_feature = feature
            sum_feature = feature
            for hop in range(self.k_hop):
                feature = feature * norm
                feature = graph.send_recv(feature, "sum")
                feature = feature * norm
                feature = (1 - self.alpha) * feature
                sum_feature += feature
            feature = sum_feature / self.k_hop + self.alpha * ori_feature

        output = self.linear(feature)
        if hasattr(self, "bias"):
            output = output + self.bias

        if self.activation is not None:
            output = self.activation(output)
        return output


class NGCFConv(nn.Layer):
    """
    Implementation of Neural Graph Collaborative Filtering (NGCF)

    This is an implementation of the paper Neural Graph Collaborative Filtering  
    (https://arxiv.org/pdf/1905.08108.pdf).

    Args:

        input_size: The size of the inputs. 

        output_size: The size of outputs

    """

    def __init__(self, input_size, output_size):
        super(NGCFConv, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        weight_attr = paddle.ParamAttr(
            initializer=nn.initializer.XavierUniform())
        bias_attr = paddle.ParamAttr(initializer=nn.initializer.XavierUniform(
            fan_in=1, fan_out=output_size))
        self.linear = nn.Linear(input_size, output_size, weight_attr,
                                bias_attr)
        self.linear2 = nn.Linear(input_size, output_size, weight_attr,
                                 bias_attr)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, graph, feature):
        """
         
        Args:
 
            graph: `pgl.Graph` instance.

            feature: A tensor with shape (num_nodes, input_size)
     
        Return:

            A tensor with shape (num_nodes, output_size)

        """
        norm = GF.degree_norm(graph)
        neigh_feature = graph.send_recv(feature, "sum")
        output = neigh_feature + feature
        output = output * norm
        output = self.linear(output) + self.linear2(feature * output)
        output = self.leaky_relu(output)
        return output


class LightGCNConv(nn.Layer):
    """
    
    Implementation of LightGCN
    
    This is an implementation of the paper LightGCN: Simplifying 
    and Powering Graph Convolution Network for Recommendation 
    (https://dl.acm.org/doi/10.1145/3397271.3401063).

    """

    def __init__(self):
        super(LightGCNConv, self).__init__()

    def forward(self, graph, feature):
        """
        Args:
 
            graph: `pgl.Graph` instance.

            feature: A tensor with shape (num_nodes, input_size)

        Return:

            A tensor with shape (num_nodes, output_size)

        """

        norm = GF.degree_norm(graph)
        feature = feature * norm
        feature = graph.send_recv(feature, "sum")
        feature = feature * norm
        return feature
