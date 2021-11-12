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
"""
    This package contains some graph transform functions.
"""

import numpy as np
import pgl


def to_undirected(graph, copy_node_feat=True, copy_edge_feat=False):
    """Convert a graph to an undirected graph.
    
    Args:

        graph (pgl.Graph): The input graph, should be in numpy format.

        copy_node_feat (bool): Whether to copy node feature in return graph. Default: True.
 
        copy_edge_feat (bool): [Alternate input] Whether to copy edge feature in return graph.

    Returns:

        g (pgl.Graph): Returns an undirected graph.

    """

    if graph.is_tensor():
        raise TypeError("The input graph should be numpy format.")

    inv_edges = np.zeros(graph.edges.shape)
    inv_edges[:, 0] = graph.edges[:, 1]
    inv_edges[:, 1] = graph.edges[:, 0]
    edges = np.vstack((graph.edges, inv_edges))
    edges = np.unique(edges, axis=0)
    g = pgl.graph.Graph(num_nodes=graph.num_nodes, edges=edges)

    if copy_node_feat:
        for k, v in graph._node_feat.items():
            g._node_feat[k] = v

    if copy_edge_feat:
        # TODO(daisiming): Support duplicate edge_feature.
        raise NotImplementedError(
            "The copy of edge feature is not implemented currently.")

    return g


def add_self_loops(graph, copy_node_feat=True, copy_edge_feat=False):
    """Add self-loops to the given graph.

    Args:

        graph (pgl.Graph): The input graph, should be in numpy format.

        copy_node_feat (bool): Whether to copy node feature in return graph. Default: True.

        copy_edge_feat (bool): [Alternate input] Whether to copy edge feature in return graph.
    
    Returns:

        g (pgl.Graph): Returns a graph with self-loops.

    """

    if graph.is_tensor():
        raise TypeError("The input graph should be numpy format.")

    self_loop_edges = np.zeros((graph.num_nodes, 2))
    self_loop_edges[:, 0] = self_loop_edges[:, 1] = np.arange(graph.num_nodes)
    edges = np.vstack((graph.edges, self_loop_edges))
    g = pgl.graph.Graph(num_nodes=graph.num_nodes, edges=edges)

    if copy_node_feat:
        for k, v in graph._node_feat.items():
            g._node_feat[k] = v

    if copy_edge_feat:
        # TODO(daisiming): Generate edge_feature of self-loops.
        raise NotImplementedError(
            "The copy of edge feature is not implemented currently.")

    return g
