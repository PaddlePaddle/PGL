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
import paddle
import paddle.nn as nn
import math
import paddle.nn.functional as F
import pgl
from pgl.utils.transform import to_dense_batch

__all__ = ["GraphMultisetTransformer"]


class MAB(nn.Layer):
    """Implementation of Multi-Head Attention

    The MH Block in paper: Accurate Learning of Graph Representations
    with Graph Multiset Pooling

    Args:
        dim_Q: The dimension of query tensor.
        dim_K: The dimension of key tensor.
        dim_V: The dimension of value tensor.
        num_heads: The number of attention heads.
        conv: The Graph Convolution Block for GMH. Default: None
        layer_norm: Whether to use LayerNorm.
    """

    def __init__(self,
                 dim_Q,
                 dim_K,
                 dim_V,
                 num_heads,
                 conv=None,
                 layer_norm=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.layer_norm = layer_norm
        self.proj_q = nn.Linear(dim_K, dim_V)
        if conv is None:
            self.layer_k = nn.Linear(dim_K, dim_V)
            self.layer_v = nn.Linear(dim_K, dim_V)
        else:
            # GCNConv is enough
            self.layer_k = conv(dim_K, dim_V)
            self.layer_v = conv(dim_K, dim_V)
        if layer_norm:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)

        self.proj_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, graph=None, mask=None):
        """
        Args:           
            Q: Query tensor with shape [-1, dim_Q]
            K: Key tensor with shape [-1, dim_K]
            graph: "Pgl.Graph" instance. Default: None
            mask: A tensor indicating the position of dummy nodes
        Return:
            A tensor with shape (-1, Q.shape[0], dim_V)
        """
        Q = self.proj_q(Q)
        if graph is not None:
            graph, x = graph
            K, V = self.layer_k(graph, x), self.layer_v(graph, x)
            K, _ = to_dense_batch(K, graph)
            V, _ = to_dense_batch(V, graph)
        else:
            K, V = self.layer_k(K), self.layer_v(K)

        # h_k = self.dim_V // self.num_heads
        Q_ = paddle.concat(Q.split(self.num_heads, 2), axis=0)
        K_ = paddle.concat(K.split(self.num_heads, 2), axis=0)
        V_ = paddle.concat(V.split(self.num_heads, 2), axis=0)
        if mask is not None:
            # mask [batch_size, seq_len]
            mask = paddle.concat([mask for _ in range(self.num_heads)], axis=0)
            mask = mask.reshape((K_.shape[0], K_.shape[1]))
            mask = mask.unsqueeze(1)
            att_score = Q_.bmm(K_.transpose([0, 2, 1]))
            att_score = att_score / math.sqrt(self.dim_V)
            A = F.softmax(mask + att_score, 1)
        else:
            A = F.softmax(
                Q_.bmm(K_.transpose([0, 2, 1])) / math.sqrt(self.dim_V), 1)
        output = paddle.concat((Q_ + A.bmm(V_)).split(self.num_heads, 0), 2)

        if self.layer_norm:
            output = self.ln0(output)
        output = output + F.relu(self.proj_o(output))
        if self.layer_norm:
            output = self.ln0(output)

        return output


"""Self-Attention Block"""


class SAB(nn.Layer):
    """Implementation of Self-Attention Block 

    The SelfAtt Block in paper: Accurate Learning of Graph Representations
    with Graph Multiset Pooling
    
    Args:
        input_dim: The size of inputs.
        output_dim. The size of outputs.
        num_heads. The number of attention heads.
        Conv: Graph Convolution Operation. Default:None
        layer_norm: Whether to use LayerNorm.
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 num_heads,
                 conv=None,
                 layer_norm=False):
        super(SAB, self).__init__()
        self.mab = MAB(input_dim,
                       input_dim,
                       output_dim,
                       num_heads,
                       conv=conv,
                       layer_norm=layer_norm)

    def forward(self, x, graph, mask):
        """
        Args:           
            x: The feature of nodes with shape [batch_size, num_nodes, feature_size]
            graph: "Pgl.Graph" instance. 
            mask: A tensor indicating the position of dummy nodes
        Return:
            A tensor with shape [batch_size, num_nodes, feature_size]
        """
        return self.mab(x, x, graph, mask)


class PMA(nn.Layer):
    """Implementation of GMPool_k Block 

    The GMPool_k Block in paper: Accurate Learning of Graph Representations
    with Graph Multiset Pooling
    
    Args:
        dim: The size of inputs.
        num_seeds: The number of nodes after pooling operation.
        num_heads: The number of attention heads.
        Conv: The Graph Convolution Block for GMH. Default: None
        layer_norm: Whether to use LayerNorm.
    """

    def __init__(self, dim, num_heads, num_seeds, conv=None, layer_norm=False):
        super(PMA, self).__init__()
        self.Q_S = self.create_parameter(
            shape=[1, num_seeds, dim],
            dtype='float32',
            default_initializer=paddle.nn.initializer.KaimingUniform())
        self.dim = dim
        self.num_seeds = num_seeds
        self.mab = MAB(dim,
                       dim,
                       dim,
                       num_heads,
                       conv=conv,
                       layer_norm=layer_norm)

    def forward(self, x, graph, mask):
        """
        Args:           
            x: The feature of nodes with shape [batch_size, num_nodes, feature_size]
            graph: "Pgl.Graph" instance. 
            mask: A tensor indicating the position of dummy nodes
        Return:
            A tensor with shape [batch_size, num_nodes, feature_size]
        """
        return self.mab(
            self.Q_S.expand([x.shape[0], self.num_seeds, self.dim]), x, graph,
            mask)


class GraphMultisetTransformer(nn.Layer):
    """Implementation of Graph Multiset Transformer pooling operator
    Referenced from: Accurate Learning of Graph Representations
    with Graph Multiset Pooling
    
    Args:
        input_dim: The size of input feature
        hidden_dim: The number of hidden units.
        output_dim: The size of output feature.
        conv: A graph convolution layer. Default: pgl.nn.GCNCONV
        num_nodes: The number of average nodes.
        pooling_ratio: The pooling ratio of GMPool. Default: 0.25
        pool_sequences: A str list that contains the blocks refered in GMT paper.
                        Default`["GMPool_G", "SelfAtt", "GMPool_I"]`)
        num_heads: The number of attention heads. Default: 4
        layer_norm: Wether to use LayerNorm. Default: False
        
    """

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 conv=None,
                 num_nodes=30,
                 pooling_ratio=0.25,
                 pool_sequences=None,
                 num_heads=4,
                 layer_norm=False):
        super(GraphMultisetTransformer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.conv = conv or pgl.nn.GCNConv
        self.num_nodes = num_nodes
        self.pooling_ratio = pooling_ratio
        self.pool_sequences = ["GMPool_G", "SelfAtt", "GMPool_I"
                               ] if pool_sequences is None else pool_sequences
        self.num_heads = num_heads
        self.layer_norm = layer_norm
        self.pools = nn.LayerList()
        num_out_nodes = math.ceil(num_nodes * pooling_ratio)
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, output_dim)

        for i, pool_type in enumerate(self.pool_sequences):
            if pool_type not in ['GMPool_G', 'GMPool_I', 'SelfAtt']:
                raise ValueError("Elements in 'pool_sequences' should be one "
                                 "of 'GMPool_G', 'GMPool_I', or 'SelfAtt'")
            if i == len(self.pool_sequences) - 1:
                num_out_nodes = 1

            if pool_type == "GMPool_G":
                self.pools.append(
                    PMA(hidden_dim,
                        num_heads,
                        num_out_nodes,
                        conv=self.conv,
                        layer_norm=layer_norm))
                num_out_nodes = math.ceil(num_out_nodes * self.pooling_ratio)
            elif pool_type == "GMPool_I":
                self.pools.append(
                    PMA(hidden_dim,
                        num_heads,
                        num_out_nodes,
                        conv=None,
                        layer_norm=layer_norm))
                num_out_nodes = math.ceil(num_out_nodes * self.pooling_ratio)
            elif pool_type == "SelfAtt":
                self.pools.append(
                    SAB(hidden_dim,
                        hidden_dim,
                        num_heads,
                        conv=None,
                        layer_norm=layer_norm))

    def forward(self, graph, x):
        """
        Args:
            graph: "pgl.Graph" instance
            x:  A tensor with shape [num_nodes, input_dim]
        Returned:
            output: the pooled output with shape[1, output_dim]
        """

        x = self.lin1(x)
        batch_x, _ = to_dense_batch(x, graph)
        mask = (batch_x.sum(-1) == 0).astype(paddle.int64).unsqueeze(0) * -1e9
        for i, (name, pool) in enumerate(zip(self.pool_sequences, self.pools)):
            g = (graph, x) if name == "GMPool_G" else None
            batch_x = pool(batch_x, g, mask)
            mask = None
        output = self.lin2(batch_x.squeeze(1))
        return output
