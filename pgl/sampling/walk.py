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
import numpy as np
from pgl import graph_kernel

__all__ = ['random_walk', 'node2vec_walk',  'node2vec_walk_plus']


def random_walk(graph, nodes, max_depth):
    """Implement of random walk.

    This function get random walks path for given nodes and depth.

    Args:
        nodes: Walk starting from nodes
        max_depth: Max walking depth

    Return:
        A list of walks.
    """
    walk_paths = []
    # init
    for node in nodes:
        walk_paths.append([node])

    cur_walk_ids = np.arange(0, len(nodes))
    cur_nodes = np.array(nodes)
    for l in range(max_depth - 1):
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

        outdegree = [len(cur_succ) for cur_succ in cur_succs]
        sample_index = np.floor(
            np.random.rand(cur_succs.shape[0]) * outdegree).astype("int64")

        nxt_cur_nodes = []
        for s, ind, walk_id in zip(cur_succs, sample_index, cur_walk_ids):
            walk_paths[walk_id].append(s[ind])
            nxt_cur_nodes.append(s[ind])
        cur_nodes = np.array(nxt_cur_nodes)
    return walk_paths


def node2vec_walk(graph, nodes, max_depth, p=1.0, q=1.0):
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
        return random_walk(graph, nodes, max_depth)

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


def node2vec_walk_plus(graph, nodes, max_depth, p=1.0, q=1.0):
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
        return random_walk(graph, nodes, max_depth)

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

        new_prev_succs = []
        for idx, (
                succ, prev_succ, walk_id, prev_node
        ) in enumerate(zip(cur_succs, prev_succs, cur_walk_ids, prev_nodes)):

            sampled_succ, new_prev_succ = graph_kernel.node2vec_plus_sample(succ, prev_succ.astype(np.int64),
                                                        prev_node, p, q)
            walk[walk_id].append(sampled_succ)
            nxt_nodes[idx] = sampled_succ
            new_prev_succs.append(new_prev_succ)

        prev_nodes = cur_nodes
        prev_succs = np.array(new_prev_succs, dtype=object)
        cur_nodes = nxt_nodes
    return walk
