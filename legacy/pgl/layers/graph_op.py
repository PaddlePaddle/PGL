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
"""This package implements common layers to help building
graph neural networks.
"""
import paddle.fluid as F 
import paddle.fluid.layers as L
from pgl import graph_wrapper
from pgl.utils import paddle_helper
from pgl.utils import op

__all__ = ['graph_pooling', 'graph_norm', 'graph_gather']


def graph_pooling(gw, node_feat, pool_type):
    """Implementation of graph pooling 

    This is an implementation of graph pooling

    Args:
        gw: Graph wrapper object (:code:`StaticGraphWrapper` or :code:`GraphWrapper`)

        node_feat: A tensor with shape (num_nodes, feature_size).

        pool_type: The type of pooling ("sum", "average" , "min")

    Return:
        A tensor with shape (num_graph, hidden_size)
    """
    graph_feat = op.nested_lod_reset(node_feat, gw.graph_lod)
    graph_feat = L.sequence_pool(graph_feat, pool_type)
    return graph_feat


def graph_norm(gw, feature):
    """Implementation of graph normalization
   
    Reference Paper: BENCHMARKING GRAPH NEURAL NETWORKS
   
    Each node features is divied by sqrt(num_nodes) per graphs.

    Args:
        gw: Graph wrapper object (:code:`StaticGraphWrapper` or :code:`GraphWrapper`)

        feature: A tensor with shape (num_nodes, hidden_size)

    Return:
        A tensor with shape (num_nodes, hidden_size)
    """
    nodes = L.fill_constant(
        [gw.num_nodes, 1], dtype="float32", value=1.0)
    norm = graph_pooling(gw, nodes, pool_type="sum")
    norm = L.sqrt(norm)
    feature_lod = op.nested_lod_reset(feature, gw.graph_lod)
    norm = L.sequence_expand_as(norm, feature_lod)
    norm.stop_gradient = True
    return feature_lod / norm


def graph_gather(gw, feature, index):
    """Implementation of graph gather 

    Gather the corresponding index for each graph.
   
    Args:
        gw: Graph wrapper object (:code:`StaticGraphWrapper` or :code:`GraphWrapper`)

        feature: A tensor with shape (num_nodes, ). 

        index (int32): A tensor with K-rank where the first dim denotes the graph.
                        Shape (num_graph, ) or (num_graph, k1, k2, k3, ..., kn).
                       WARNING: We dont support negative index.

    Return:
        A tensor with shape (num_graph, k1, k2, k3, ..., kn, hidden_size)
    """
    shape = L.shape(index)
    output_dim = int(feature.shape[-1])
    index = index + gw.graph_lod[:-1]
    index = L.reshape(index, [-1])
    feature = L.gather(feature, index, overwrite=False)
    new_shape = []
    for i in range(shape.shape[0]):
        new_shape.append(shape[i])
    new_shape.append(output_dim)
    feature = L.reshape(feature, new_shape)
    return feature

