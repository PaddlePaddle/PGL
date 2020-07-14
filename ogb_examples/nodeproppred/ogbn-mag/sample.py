# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import copy
import numpy as np

from pgl.sample import deepwalk_sample, traverse
from pgl.sample import extract_edges_from_nodes


def graph_saint_random_walk_sample(graph,
                                   hetergraph,
                                   nodes,
                                   max_depth,
                                   alias_name=None,
                                   events_name=None):
    """Implement of graph saint random walk sample for hetergraph.

    Args:
        graph: A pgl graph instance, homograph for hetergraph
        hetergraph:  A pgl hetergraph instance
        nodes: Walk starting from nodes
        max_depth: Max walking depth

    Return:
        a subgraph of sampled nodes.
    """
    # the seed for numpy should be reset in multiprocessing.
    np.random.seed()
    graph.outdegree()
    # sample random nodes
    nodes = np.random.choice(
        np.arange(
            graph.num_nodes, dtype='int64'), size=20000, replace=False)
    walks = deepwalk_sample(graph, nodes, max_depth, alias_name, events_name)
    sample_nodes = []
    for walk in walks:
        sample_nodes.extend(walk)

    sample_nodes = np.unique(sample_nodes)
    eids = extract_edges_from_nodes(hetergraph, sample_nodes)
    subgraph = hetergraph.subgraph(
        nodes=sample_nodes, eid=eids, with_node_feat=True, with_edge_feat=True)

    return subgraph, sample_nodes


def graph_saint_hetero(graph, hetergraph, batch_nodes, max_depth=2):
    subgraph, sample_nodes = graph_saint_random_walk_sample(
        graph, hetergraph, batch_nodes, max_depth)
    # the new index of sample_nodes is range(0, len(sample_nodes))
    all_label = graph._node_feat['train_label'][sample_nodes]
    train_index = np.where(all_label > -1)[0]
    train_label = all_label[train_index]
    return subgraph, train_index, sample_nodes, train_label


def k_hop_sampler(graph, hetergraph, batch_nodes, samples=[30, 30]):
    # the seed for numpy should be reset in multiprocessing.
    np.random.seed()
    start_nodes = copy.deepcopy(batch_nodes)
    nodes = start_nodes
    edges = []
    for max_deg in samples:
        pred_nodes = graph.sample_predecessor(start_nodes, max_degree=max_deg)
        for dst_node, src_nodes in zip(start_nodes, pred_nodes):
            for src_node in src_nodes:
                edges.append((src_node, dst_node))
        last_nodes = nodes
        nodes = [nodes, pred_nodes]
        nodes = list(set(traverse(nodes)))
        # Find new nodes
        start_nodes = list(set(nodes) - set(last_nodes))
        if len(start_nodes) == 0:
            break

    # TODO: Only use certrain sampled edges to construct subgraph. 
    nodes = np.unique(np.array(nodes, dtype='int64'))
    eids = extract_edges_from_nodes(hetergraph, nodes)

    subgraph = hetergraph.subgraph(
        nodes=nodes, eid=eids, with_node_feat=True, with_edge_feat=True)
    train_index = subgraph.reindex_from_parrent_nodes(batch_nodes)

    return subgraph, train_index, nodes, None
