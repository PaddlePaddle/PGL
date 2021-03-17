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
"""
    This package implement graph sampling algorithm.
"""
import time
import copy

import numpy as np
import pgl
from pgl.utils.logger import log
from pgl import graph_kernel

__all__ = [
    'graphsage_sample', 'node2vec_sample', 'deepwalk_sample',
    'metapath_randomwalk', 'pinsage_sample', 'graph_saint_random_walk_sample'
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

    subgraphs = []
    for i in range(num_layers):
        subgraphs.append(
            graph.subgraph(
                nodes=layer_nodes[0], eid=layer_eids[i], edges=layer_edges[i]))
        # only for this task
        subgraphs[i].node_feat["index"] = np.array(
            layer_nodes[0], dtype="int64")

    return subgraphs


def alias_sample(size, alias, events):
    """Implement of alias sample.
    Args:
        size: Output shape.
        alias: The alias table build by `alias_sample_build_table`.
        events: The events table build by `alias_sample_build_table`.

    Return:
        samples: The generated random samples.
    """
    rand_num = np.random.uniform(0.0, len(alias), size)
    idx = rand_num.astype("int64")
    uni = rand_num - idx
    flags = (uni >= alias[idx])
    idx[flags] = events[idx][flags]
    return idx


def graph_alias_sample_table(graph, edge_weight_name):
    """Build alias sample table for weighted deepwalk.
    Args:
        graph: The input graph
        edge_weight_name: The name of edge weight in edge_feat.

    Return:
        Alias sample tables for each nodes.
    """
    edge_weight = graph.edge_feat[edge_weight_name]
    _, eids_array = graph.successor(return_eids=True)
    alias_array, events_array = [], []
    for eids in eids_array:
        probs = edge_weight[eids]
        probs /= np.sum(probs)
        alias, events = graph_kernel.alias_sample_build_table(probs)
        alias_array.append(alias), events_array.append(events)
    alias_array, events_array = np.array(alias_array), np.array(events_array)
    return alias_array, events_array


def deepwalk_sample(graph, nodes, max_depth, alias_name=None,
                    events_name=None):
    """Implement of random walk.

    This function get random walks path for given nodes and depth.

    Args:
        nodes: Walk starting from nodes
        max_depth: Max walking depth

    Return:
        A list of walks.
    """
    walk = []
    # init
    for node in nodes:
        walk.append([node])

    cur_walk_ids = np.arange(0, len(nodes))
    cur_nodes = np.array(nodes)
    for l in range(max_depth):
        # select the walks not end
        cur_succs = graph.successor(cur_nodes)
        mask = [len(succ) > 0 for succ in cur_succs]

        if np.any(mask):
            cur_walk_ids = cur_walk_ids[mask]
            cur_nodes = cur_nodes[mask]
            cur_succs = cur_succs[mask]
        else:
            # stop when all nodes have no successor
            break

        if alias_name is not None and events_name is not None:
            sample_index = [
                alias_sample([1], graph.node_feat[alias_name][node],
                             graph.node_feat[events_name][node])[0]
                for node in cur_nodes
            ]
        else:
            outdegree = [len(cur_succ) for cur_succ in cur_succs]
            sample_index = np.floor(
                np.random.rand(cur_succs.shape[0]) * outdegree).astype("int64")

        nxt_cur_nodes = []
        for s, ind, walk_id in zip(cur_succs, sample_index, cur_walk_ids):
            walk[walk_id].append(s[ind])
            nxt_cur_nodes.append(s[ind])
        cur_nodes = np.array(nxt_cur_nodes)
    return walk


def node2vec_sample(graph, nodes, max_depth, p=1.0, q=1.0):
    """Implement of node2vec random walk.

    Reference paper: https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf.

    Args:
        graph: A pgl graph instance
        nodes: Walk starting from nodes
        max_depth: Max walking depth
        p: Return parameter
        q: In-out parameter

    Return:
        A list of walks.
    """
    if p == 1.0 and q == 1.0:
        return deepwalk_sample(graph, nodes, max_depth)

    walk = []
    # init
    for node in nodes:
        walk.append([node])

    cur_walk_ids = np.arange(0, len(nodes))
    cur_nodes = np.array(nodes)
    prev_nodes = np.array([-1] * len(nodes), dtype="int64")
    prev_succs = np.array([[]] * len(nodes), dtype="int64")
    for l in range(max_depth):
        # select the walks not end
        cur_succs = graph.successor(cur_nodes)

        mask = [len(succ) > 0 for succ in cur_succs]
        if np.any(mask):
            cur_walk_ids = cur_walk_ids[mask]
            cur_nodes = cur_nodes[mask]
            prev_nodes = prev_nodes[mask]
            prev_succs = prev_succs[mask]
            cur_succs = cur_succs[mask]
        else:
            # stop when all nodes have no successor
            break
        num_nodes = cur_nodes.shape[0]
        nxt_nodes = np.zeros(num_nodes, dtype="int64")

        for idx, (
                succ, prev_succ, walk_id, prev_node
        ) in enumerate(zip(cur_succs, prev_succs, cur_walk_ids, prev_nodes)):

            sampled_succ = graph_kernel.node2vec_sample(succ, prev_succ,
                                                        prev_node, p, q)
            walk[walk_id].append(sampled_succ)
            nxt_nodes[idx] = sampled_succ

        prev_nodes, prev_succs = cur_nodes, cur_succs
        cur_nodes = nxt_nodes
    return walk


def metapath_randomwalk(graph,
                        start_nodes,
                        metapath,
                        walk_length,
                        alias_name=None,
                        events_name=None):
    """Implementation of metapath random walk in heterogeneous graph.

    Args:
        graph: instance of pgl heterogeneous graph
        start_nodes: start nodes to generate walk
        metapath: meta path for sample nodes.
            e.g: "c2p-p2a-a2p-p2c"
        walk_length: the walk length

    Return:
        a list of metapath walks.

    """

    edge_types = metapath.split('-')

    walk = []
    for node in start_nodes:
        walk.append([node])

    cur_walk_ids = np.arange(0, len(start_nodes))
    cur_nodes = np.array(start_nodes)
    mp_len = len(edge_types)
    for i in range(0, walk_length - 1):
        g = graph[edge_types[i % mp_len]]

        cur_succs = g.successor(cur_nodes)
        mask = [len(succ) > 0 for succ in cur_succs]

        if np.any(mask):
            cur_walk_ids = cur_walk_ids[mask]
            cur_nodes = cur_nodes[mask]
            cur_succs = cur_succs[mask]
        else:
            # stop when all nodes have no successor
            break

        if alias_name is not None and events_name is not None:
            sample_index = [
                alias_sample([1], g.node_feat[alias_name][node],
                             g.node_feat[events_name][node])[0]
                for node in cur_nodes
            ]
        else:
            outdegree = [len(cur_succ) for cur_succ in cur_succs]
            sample_index = np.floor(
                np.random.rand(cur_succs.shape[0]) * outdegree).astype("int64")

        nxt_cur_nodes = []
        for s, ind, walk_id in zip(cur_succs, sample_index, cur_walk_ids):
            walk[walk_id].append(s[ind])
            nxt_cur_nodes.append(s[ind])
        cur_nodes = np.array(nxt_cur_nodes)

    return walk


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
                   ignore_edges=set()):
    """Implement of graphsage sample.

    Reference paper: .

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
    start = time.time()
    num_layers = len(samples)
    start_nodes = nodes
    edges, weights = [], []
    layer_nodes, layer_edges, layer_weights = [], [], []
    ignore_edge_set = set([edge_hash(src, dst) for src, dst in ignore_edges])

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
        start = time.time()

    feed_dict = {}

    subgraphs = []

    for i in range(num_layers):
        edge_feat_dict = {
            "weight": np.array(
                layer_weights[i], dtype='float32')
        }
        subgraphs.append(
            graph.subgraph(
                nodes=layer_nodes[0],
                edges=layer_edges[i],
                edge_feats=edge_feat_dict,
                with_edge_feat=False))
        subgraphs[i].node_feat["index"] = np.array(
            layer_nodes[0], dtype="int64")

    return subgraphs


def extract_edges_from_nodes(graph, sample_nodes):
    eids = graph_kernel.extract_edges_from_nodes(
        graph.adj_src_index._indptr, graph.adj_src_index._sorted_v,
        graph.adj_src_index._sorted_eid, sample_nodes)
    return eids


def graph_saint_random_walk_sample(graph,
                                   nodes,
                                   max_depth,
                                   alias_name=None,
                                   events_name=None):
    """Implement of graph saint random walk sample.

    First, this function will get random walks path for given nodes and depth.
    Then, it will create subgraph from all sampled nodes.

    Reference Paper: https://arxiv.org/abs/1907.04931

    Args:
        graph: A pgl graph instance
        nodes: Walk starting from nodes
        max_depth: Max walking depth

    Return:
        a subgraph of sampled nodes.
    """
    graph.outdegree()
    walks = deepwalk_sample(graph, nodes, max_depth, alias_name, events_name)
    sample_nodes = []
    for walk in walks:
        sample_nodes.extend(walk)
    sample_nodes = np.unique(sample_nodes)
    eids = extract_edges_from_nodes(graph, sample_nodes)
    subgraph = graph.subgraph(
        nodes=sample_nodes, eid=eids, with_node_feat=True, with_edge_feat=True)
    subgraph.node_feat["index"] = np.array(sample_nodes, dtype="int64")
    return subgraph


def edge_drop(graph_wrapper, dropout_rate, keep_self_loop=True):
    if dropout_rate < 1e-5:
        return graph_wrapper
    else:
        return pgl.graph_wrapper.DropEdgeWrapper(graph_wrapper,
                   dropout_rate,
                   keep_self_loop)
