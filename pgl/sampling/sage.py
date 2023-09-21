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
import paddle
import pgl
from pgl import graph_kernel
from pgl.sampling.custom import subgraph
from pgl.utils.logger import log
from pgl.utils.helper import to_paddle_tensor

__all__ = [
    'graphsage_sample',
    'NeighborSampler',
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


class NeighborSampler(object):
    """ GPU Sampler for homogeneous graph """

    def __init__(self, graph, samples, uva=False):
        self.graph = graph
        self.samples = samples
        self.row = to_paddle_tensor(self.graph.adj_dst_index._sorted_v, uva)
        self.colptr = to_paddle_tensor(self.graph.adj_dst_index._indptr, uva)

    def sample_neighbors(self, nodes):
        graph_list = []
        for size in self.samples:
            # If you want to sample with faster speed, you can refer to
            # the corresponding geometric API documents for modification.
            neighbors, neighbors_count = paddle.geometric.sample_neighbors(
                self.row, self.colptr, nodes, sample_size=size)
            edge_src, edge_dst, sample_index = paddle.geometric.reindex_graph(
                nodes, neighbors, neighbors_count)
            subgraph = pgl.Graph(
                num_nodes=sample_index.shape[0],
                edges=paddle.concat(
                    [edge_src.reshape([-1, 1]), edge_dst.reshape([-1, 1])],
                    axis=-1))
            graph_list.append((subgraph, nodes.shape[0]))
            nodes = sample_index
        return graph_list[::-1], nodes


class HeteroNeighborSampler(object):
    """ GPU Sampler for heterogeneous graph """

    def __init__(self, graph_list, sample_list, uva=False):
        raise NotImplementedError
