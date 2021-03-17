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

import pgl
import numpy as np
from pgl.graph import Graph

__all__ = []
__all__.append("subgraph")


def subgraph(graph,
             nodes,
             eid=None,
             edges=None,
             with_node_feat=True,
             with_edge_feat=True):
    """Generate subgraph with nodes and edge ids.
    This function will generate a :code:`pgl.graph.Subgraph` object and
    copy all corresponding node and edge features. Nodes and edges will
    be reindex from 0. Eid and edges can't both be None.
    WARNING: ALL NODES IN EID MUST BE INCLUDED BY NODES

    Args:
        nodes: Node ids which will be included in the subgraph.
        eid (optional): Edge ids which will be included in the subgraph.
        edges (optional): Edge(src, dst) list which will be included in the subgraph.

        with_node_feat: Whether to inherit node features from parent graph.
        with_edge_feat: Whether to inherit edge features from parent graph.

    Return:
        A :code:`pgl.Graph` object.
    """
    assert not graph.is_tensor(), "You must call Graph.numpy() first."

    if eid is None and edges is None:
        raise ValueError("Eid and edges can't be None at the same time.")

    reindex = {}

    for ind, node in enumerate(nodes):
        reindex[node] = ind

    sub_edge_feat = {}
    if edges is None:
        edges = graph._edges[eid]
    else:
        edges = np.array(edges, dtype="int64")

    if with_edge_feat:
        for key, value in graph._edge_feat.items():
            if eid is None:
                raise ValueError("Eid can not be None with edge features.")
            sub_edge_feat[key] = value[eid]

    sub_edges = pgl.graph_kernel.map_edges(
        np.arange(
            len(edges), dtype="int64"), edges, reindex)

    sub_node_feat = {}
    if with_node_feat:
        for key, value in graph._node_feat.items():
            sub_node_feat[key] = value[nodes]

    g = Graph(
        edges=sub_edges,
        num_nodes=len(nodes),
        node_feat=sub_node_feat,
        edge_feat=sub_edge_feat)

    return g
