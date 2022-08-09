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
import paddle
from pgl.math import segment_sum
from pgl.utils.helper import maybe_num_nodes


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


def to_dense_batch(x, graph, fill_value=0, max_num_nodes=None):
    """Transfrom a batch of graphs to a dense node feature tensor and 
       provide the mask  holing the positions of dummy nodes

    Args:
        x (paddle.tensor): The feature map of nodes
        
        graph (pgl.Graph): The graph holing the graph node id

        fill_value (bool): The value of dummy nodes. Default: 0.

        max_node_nodes: The dimension of nodes in dense batch. Default: None
    
    Returns:

        out (paddle.tensor): Returns a dense node feature tensor (shape = [batch_size,max_num_nodes,-1])
        mask (paddle.tensor): Return a mask indicating the position of dummy nodes (shape = [batch_size, max_num_nodes])

    """
    graph_node_id = graph.graph_node_id
    batch_size = (graph_node_id.max().item()) + 1
    num_nodes = segment_sum(paddle.ones([x.shape[0]]), graph_node_id)
    cum_nodes = paddle.concat([paddle.zeros([1]), num_nodes.cumsum(0)])
    if max_num_nodes is None:
        max_num_nodes = int(num_nodes.max())
    idx = paddle.arange(graph_node_id.shape[0], dtype=paddle.int64)
    idx = (idx - cum_nodes[graph_node_id]) + (graph_node_id * max_num_nodes)
    size = [batch_size * max_num_nodes] + list(x.shape)[1:]
    out = paddle.full(size, fill_value)
    out = paddle.scatter(out, idx, x)
    out = out.reshape([batch_size, max_num_nodes] + list(x.shape)[1:])
    mask = paddle.ones([batch_size * max_num_nodes])
    mask = paddle.scatter(mask, idx, paddle.zeros([idx.shape[0]])).reshape(
        [batch_size, max_num_nodes]).astype(paddle.bool)
    return out, mask


def filter_adj(edge_index, perm, edge_attr=None, num_nodes=None):
    """Accoding to the reserved nodes,  updating edges in graph and reindex nodes

    Args:
        edge_index (paddle.tensor): The edge_index of graph (shape: [|E|, 2 ])
        
        perm (paddle.tensor): The index of reserved nodes

        edge_attr (edge_attr): The attribute of edges. Default: None.

        num_nodes: The number of nodes.  Default: None
    
    Returns:

        out (paddle.tensor): Returns a udpated edge index tensor (shape = [|E|, 2'])
        edge_attr (paddle.tensor): Return a edge attribute tensor if input edge_attr is not None

    """
    num_nodes = maybe_num_nodes(edge_index)
    mask = paddle.full([num_nodes], -1, dtype=paddle.float32)
    i = paddle.arange(perm.shape[0], dtype=paddle.float32)
    mask = paddle.scatter(mask, perm, i)
    mask = mask.astype(paddle.int64)
    row, col = edge_index.transpose([1, 0])
    row, col = mask[row], mask[col]
    mask = (row >= 0).logical_and((col >= 0))
    row, col = row[mask.nonzero()].reshape([-1]), col[mask.nonzero()].reshape(
        [-1])
    if edge_attr is not None:
        edge_attr = edge_attr[mask]
    return paddle.stack([row, col], 0).transpose([1, 0]), edge_attr
