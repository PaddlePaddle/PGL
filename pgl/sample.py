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
    'metapath_randomwalk'
]


def edge_hash(src, dst):
    """edge_hash
    """
    return src * 100000007 + dst


def graphsage_sample(graph, nodes, samples, ignore_edges=[]):
    """Implement of graphsage sample.
    
    Reference paper: https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf.

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
        log.debug("sample_predecessor time: %s" % (time.time() - start))
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
        log.debug("flat time: %s" % (time.time() - start))
        start = time.time()
        # Find new nodes

    feed_dict = {}

    subgraphs = []
    for i in range(num_layers):
        subgraphs.append(
            graph.subgraph(
                nodes=layer_nodes[0], eid=layer_eids[i], edges=layer_edges[i]))
        # only for this task
        subgraphs[i].node_feat["index"] = np.array(
            layer_nodes[0], dtype="int64")
    log.debug("subgraph time: %s" % (time.time() - start))

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
