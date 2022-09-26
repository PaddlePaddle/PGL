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
"""
    This package implement graph sampling algorithm.
"""
import time
import copy

import numpy as np
from pgl import graph_kernel

from pgl.sampling.custom import subgraph

__all__ = [
    'graphsage_sample',
    'pinsage_sample',
]


def traverse(item):
    """traverse the list or numpy"""
    if isinstance(item, list) or isinstance(item, np.ndarray):
        for i in iter(item):
            for j in traverse(i):
                yield j
    else:
        yield item


def flat_node_and_edge(nodes, eids, weights=None):
    """flatten the sub-lists to one list"""
    nodes = list(set(traverse(nodes)))
    eids = list(traverse(eids))
    if weights is not None:
        weights = list(traverse(weights))
    return nodes, eids, weights


def edge_hash(src, dst):
    """edge_hash
    """
    return src * 100000007 + dst


def graphsage_sample(graph, nodes, samples, ignore_edges=[]):
    """Implement of graphsage sample.
    Reference paper: https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf.
    Args:
        graph: A pgl graph instance
        nodes: Sample starting from nodes
        samples: A list, number of neighbors in each layer
        ignore_edges: list of edge(src, dst) will be ignored.
    Return:
        A list of subgraphs
    """
    assert not graph.is_tensor(), "You must call Graph.numpy() first."
    node_index = copy.deepcopy(nodes)
    start = time.time()
    num_layers = len(samples)
    start_nodes = nodes
    nodes = list(start_nodes)
    eids, edges = [], []
    nodes_set = set(nodes)
    layer_nodes, layer_eids, layer_edges = [], [], []
    ignore_edge_set = set([edge_hash(src, dst) for src, dst in ignore_edges])

    for layer_idx in reversed(range(num_layers)):
        if len(start_nodes) == 0:
            layer_nodes = [nodes] + layer_nodes
            layer_eids = [eids] + layer_eids
            layer_edges = [edges] + layer_edges
            continue
        batch_pred_nodes, batch_pred_eids = graph.sample_predecessor(
            start_nodes, samples[layer_idx], return_eids=True)
        start = time.time()
        last_nodes_set = nodes_set

        nodes, eids = copy.copy(nodes), copy.copy(eids)
        edges = copy.copy(edges)
        nodes_set, eids_set = set(nodes), set(eids)
        for srcs, dst, pred_eids in zip(batch_pred_nodes, start_nodes,
                                        batch_pred_eids):
            for src, eid in zip(srcs, pred_eids):
                if edge_hash(src, dst) in ignore_edge_set:
                    continue
                if eid not in eids_set:
                    eids.append(eid)
                    edges.append([src, dst])
                    eids_set.add(eid)
                if src not in nodes_set:
                    nodes.append(src)
                    nodes_set.add(src)
        layer_edges = [edges] + layer_edges
        start_nodes = list(nodes_set - last_nodes_set)
        layer_nodes = [nodes] + layer_nodes
        layer_eids = [eids] + layer_eids
        start = time.time()
        # Find new nodes

    from_reindex = {x: i for i, x in enumerate(layer_nodes[0])}
    node_index = graph_kernel.map_nodes(node_index, from_reindex)
    sample_index = np.array(layer_nodes[0], dtype="int64")

    graph_list = []
    for i in range(num_layers):
        sg = subgraph(
            graph,
            nodes=layer_nodes[0],
            eid=layer_eids[i],
            edges=layer_edges[i])
        graph_list.append((sg, sample_index, node_index))

    return graph_list


def random_walk_with_start_prob(graph, nodes, max_depth, proba=0.5):
    """Implement of random walk with the probability of returning the origin node.

    This function get random walks path for given nodes and depth.

    Args:
        nodes: Walk starting from nodes
        max_depth: Max walking depth
        proba: the proba to return the origin node

    Return:
        A list of walks.
    """
    walk = []
    # init
    for node in nodes:
        walk.append([node])

    walk_ids = np.arange(0, len(nodes))
    cur_nodes = np.array(nodes)
    nodes = np.array(nodes)
    for l in range(max_depth):
        # select the walks not end
        if l >= 1:
            return_proba = np.random.rand(cur_nodes.shape[0])
            proba_mask = (return_proba < proba)
            cur_nodes[proba_mask] = nodes[proba_mask]
        outdegree = graph.outdegree(cur_nodes)
        mask = (outdegree != 0)
        if np.any(mask):
            cur_walk_ids = walk_ids[mask]
            outdegree = outdegree[mask]
        else:
            # stop when all nodes have no successor, wait start next loop to get precesssor
            continue
        succ = graph.successor(cur_nodes[mask])
        sample_index = np.floor(
            np.random.rand(outdegree.shape[0]) * outdegree).astype("int64")

        nxt_cur_nodes = cur_nodes
        for s, ind, walk_id in zip(succ, sample_index, cur_walk_ids):
            walk[walk_id].append(s[ind])
            nxt_cur_nodes[walk_id] = s[ind]
        cur_nodes = np.array(nxt_cur_nodes)
    return walk


def pinsage_sample(graph,
                   nodes,
                   samples,
                   top_k=10,
                   proba=0.5,
                   norm_bais=1.0,
                   ignore_edges=None):
    """Implement of graphsage sample.

    Reference paper: https://arxiv.org/abs/1806.01973

    Args:
        graph: A pgl graph instance
        nodes: Sample starting from nodes
        samples: A list, number of neighbors in each layer
        top_k: select the top_k visit count nodes to construct the edges
        proba: the probability to return the origin node
        norm_bais: the normlization for the visit count
        ignore_edges: list of edge(src, dst) will be ignored.

    Return:
        A list of subgraphs
    """
    if ignore_edges is None:
        ignore_edges = set()
    num_layers = len(samples)
    start_nodes = nodes
    node_index = copy.deepcopy(nodes)
    edges, weights = [], []
    layer_nodes, layer_edges, layer_weights = [], [], []
    ignore_edge_set = {edge_hash(src, dst) for src, dst in ignore_edges}

    for layer_idx in reversed(range(num_layers)):
        if len(start_nodes) == 0:
            layer_nodes = [nodes] + layer_nodes
            layer_edges = [edges] + layer_edges
            layer_edges_weight = [weights] + layer_weights
            continue
        walks = random_walk_with_start_prob(
            graph, start_nodes, samples[layer_idx], proba=proba)
        walks = [walk[1:] for walk in walks]
        pred_edges = []
        pred_weights = []
        pred_nodes = []
        for node, walk in zip(start_nodes, walks):
            walk_nodes = []
            walk_weights = []
            count_sum = 0

            for random_walk_node in walk:
                if len(ignore_edge_set) > 0 and random_walk_node != node and \
                    edge_hash(random_walk_node, node) in ignore_edge_set:
                    continue
                walk_nodes.append(random_walk_node)
            unique, counts = np.unique(walk_nodes, return_counts=True)
            frequencies = np.asarray((unique, counts)).T
            frequencies = frequencies[np.argsort(frequencies[:, 1])]
            frequencies = frequencies[-1 * top_k:, :]
            for random_walk_node, random_count in zip(
                    frequencies[:, 0].tolist(), frequencies[:, 1].tolist()):
                pred_nodes.append(random_walk_node)
                pred_edges.append((random_walk_node, node))
                walk_weights.append(random_count)
                count_sum += random_count
            count_sum += len(walk_weights) * norm_bais
            walk_weights = (np.array(walk_weights) + norm_bais) / (count_sum)
            pred_weights.extend(walk_weights.tolist())
        last_node_set = set(nodes)
        nodes, edges, weights = flat_node_and_edge([nodes, pred_nodes], \
            [edges, pred_edges], [weights, pred_weights])

        layer_edges = [edges] + layer_edges
        layer_weights = [weights] + layer_weights
        layer_nodes = [nodes] + layer_nodes

        start_nodes = list(set(nodes) - last_node_set)

    from_reindex = {x: i for i, x in enumerate(layer_nodes[0])}
    node_index = graph_kernel.map_nodes(node_index, from_reindex)
    sample_index = np.array(layer_nodes[0], dtype="int64")

    subgraphs = []

    for i in range(num_layers):
        edge_feat_dict = {
            "weight": np.array(
                layer_weights[i], dtype='float32').rshape([-1, 1])
        }
        sg = subgraph(
            graph,
            nodes=layer_nodes[0],
            edges=layer_edges[i],
            append_edges_feat=edge_feat_dict,
            with_edge_feat=False)
        subgraphs.append((sg, sample_index, node_index))

    return subgraphs
