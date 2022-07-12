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
"""This package implements common pooling to help building
graph neural networks.
"""

import warnings
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import pgl
import pgl.math as math

__all__ = ["GraphPool", "GraphNorm", "Set2Set", "GlobalAttention"]


class GraphPool(nn.Layer):
    """Implementation of graph pooling

    This is an implementation of graph pooling

    Args:
        pool_type: The type of pooling ("sum", "mean" , "min", "max"). Default:None

    """

    def __init__(self, pool_type=None):
        super(GraphPool, self).__init__()
        self.pool_type = pool_type

    def forward(self, graph, feature, pool_type=None):
        """
         Args:
            graph: the graph object from (:code:`Graph`)

            feature: A tensor with shape (num_nodes, feature_size).

            pool_type: The type of pooling ("sum", "mean" , "min", "max"). Default:None
        Return:
            A tensor with shape (num_graph, feature_size)
        """
        if pool_type is not None:
            warnings.warn("The pool_type (%s) argument in forward function " \
                    "will be discarded in the future, " \
                    "please initialize it when creating a GraphPool instance.")
        else:
            pool_type = self.pool_type
        graph_feat = math.segment_pool(feature, graph.graph_node_id, pool_type)
        return graph_feat


class GraphNorm(nn.Layer):
    """Implementation of graph normalization
   
    Reference Paper: BENCHMARKING GRAPH NEURAL NETWORKS
    
    Each node features is divied by sqrt(num_nodes) per graphs.
    
    """

    def __init__(self):
        super(GraphNorm, self).__init__()
        self.graph_pool = GraphPool(pool_type="sum")

    def forward(self, graph, feature):
        """Forward function of GraphNorm

        Args:
            graph: the graph object from (:code:`Graph`)

            feature: A tensor with shape (num_nodes, feature_size).

        Return:
            A tensor with shape (num_nodes, hidden_size)
        """
        nodes = paddle.ones(shape=[graph.num_nodes, 1], dtype="float32")
        norm = self.graph_pool(graph, nodes)
        norm = paddle.sqrt(norm)
        norm = paddle.gather(norm, graph.graph_node_id)
        return feature / norm


class Set2Set(nn.Layer):
    """Implementation of Graph Global Pooling "Set2Set".
    
    Reference Paper: ORDER MATTERS: SEQUENCE TO SEQUENCE 

    Args:
        input_dim (int): dimentional size of input
        n_iters: number of iteration
        n_layers: number of LSTM layers
    Return:
        output_feat: output feature of set2set pooling with shape [batch, 2*dim].
    """

    def __init__(self, input_dim, n_iters, n_layers=1):
        super(Set2Set, self).__init__()
        self.input_dim = input_dim
        self.output_dim = 2 * input_dim
        self.n_iters = n_iters
        self.n_layers = n_layers
        self.lstm = paddle.nn.LSTM(
            input_size=self.output_dim,
            hidden_size=self.input_dim,
            num_layers=n_layers,
            time_major=True)

    def forward(self, graph, x):
        """Forward function of Graph Global Pooling "Set2Set".
        
        Args:
            graph: the graph object from (:code:`Graph`)
            x: A tensor with shape (num_nodes, feature_size).
        Return:
            output_feat: A tensor with shape (num_nodes, output_size).
        """
        graph_id = graph.graph_node_id
        batch_size = graph_id.max() + 1
        h = (
            paddle.zeros((self.n_layers, batch_size, self.input_dim)),
            paddle.zeros((self.n_layers, batch_size, self.input_dim)), )
        q_star = paddle.zeros((batch_size, self.output_dim))
        for _ in range(self.n_iters):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.reshape((batch_size, self.input_dim))
            e = (x * q.index_select(
                graph_id, axis=0)).sum(axis=-1, keepdim=True)
            a = math.segment_softmax(e, graph_id)
            r = math.segment_sum(a * x, graph_id)
            q_start = paddle.concat([q, r], axis=-1)

        return q_start


class GlobalAttention(nn.Layer):
    """Implementation of Graph Global Pooling "GlobalAttention"
        
    Reference Paper: Gated Graph Sequence Neural Networks.

    Args:
        gate: a neural network that mapping input \in [-1, channels] to output \in [-1, 1]
        nn: a neural network that mapping input \in [-1, in_channels] to output \in [-1, out_channels]. Default:None
    """

    def __init__(self, gate, nn=None):
        super(GlobalAttention, self).__init__()
        self.gate = gate
        self.nn = nn

    def forward(self, graph, x):
        """
        Args:
            graph: the graph object from (:code:`Graph`)
            x: A tensor with shape (num_nodes, feature_size).
            
        Return:
            output_feat: A tensor with shape (num_nodes, output_size).
        """

        graph_id = graph.graph_node_id
        gate_x = self.gate(x).reshape(shape=(-1, 1))
        x = self.nn(x) if self.nn else x
        assert x.ndim == gate_x.ndim and x.shape[0] == gate_x.shape[0]
        gate_x = math.segment_softmax(gate_x, graph_id)
        output = math.segment_sum(gate_x * x, graph_id)
        return output
