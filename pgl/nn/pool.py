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

__all__ = ["GraphPool", "GraphNorm"]


class GraphPool(nn.Layer):
    """Implementation of graph pooling

    This is an implementation of graph pooling

    Args:
        graph: the graph object from (:code:`Graph`)

        feature: A tensor with shape (num_nodes, feature_size).

        pool_type: The type of pooling ("sum", "mean" , "min", "max")

    Return:
        A tensor with shape (num_graph, feature_size)
    """

    def __init__(self, pool_type=None):
        super(GraphPool, self).__init__()
        self.pool_type = pool_type

    def forward(self, graph, feature, pool_type=None):
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

    Args:
        graph: the graph object from (:code:`Graph`)

        feature: A tensor with shape (num_nodes, feature_size).

    Return:
        A tensor with shape (num_nodes, hidden_size)
    """

    def __init__(self):
        super(GraphNorm, self).__init__()
        self.graph_pool = GraphPool(pool_type="sum")

    def forward(self, graph, feature):
        nodes = paddle.ones(shape=[graph.num_nodes, 1], dtype="float32")
        norm = self.graph_pool(graph, nodes)
        norm = paddle.sqrt(norm)
        norm = paddle.gather(norm, graph.graph_node_id)
        return feature / norm
