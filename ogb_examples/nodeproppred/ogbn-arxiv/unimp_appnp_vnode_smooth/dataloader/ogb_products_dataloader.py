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
'''
ogb_products_dataloader
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

from dataloader.base_dataloader import BaseDataGenerator
from pgl.contrib.ogb.nodeproppred.dataset_pgl import PglNodePropPredDataset

import tqdm
from collections import namedtuple
import pgl
import numpy as np
import copy

def add_self_loop_for_subgraph(graph): 
    '''add_self_loop_for_subgraph
    '''
    self_loop_edges = np.zeros((graph.num_nodes, 2))
    self_loop_edges[:, 0] = self_loop_edges[:, 1] = np.arange(graph.num_nodes)
    edges = np.vstack((graph.edges, self_loop_edges))
    edges = np.unique(edges, axis=0)
    g = pgl.graph.SubGraph(num_nodes=graph.num_nodes, edges=edges, reindex=graph._from_reindex)
    for k, v in graph._node_feat.items():
        g._node_feat[k] = v
    return graph


def traverse(item):
    """traverse
    """
    if isinstance(item, list) or isinstance(item, np.ndarray):
        for i in iter(item):
            for j in traverse(i):
                yield j
    else:
        yield item


def flat_node_and_edge(nodes):
    """flat_node_and_edge
    """
    nodes = list(set(traverse(nodes)))
    return nodes


def k_hop_sampler(graph, samples, batch_nodes):
    graph_list = []
    for max_deg in samples:
        start_nodes = copy.deepcopy(batch_nodes)
        edges = []
        if max_deg == -1:
            pred_nodes = graph.predecessor(start_nodes)
        else:
            pred_nodes = graph.sample_predecessor(start_nodes, max_degree=max_deg)

        for dst_node, src_nodes in zip(start_nodes, pred_nodes):
            for src_node in src_nodes:
                edges.append((src_node, dst_node))

        nodes = [start_nodes, pred_nodes]
        nodes = flat_node_and_edge(nodes)
        
        subgraph = graph.subgraph(
        nodes=nodes, edges=edges, with_node_feat=False, with_edge_feat=False)
        subgraph = add_self_loop_for_subgraph(subgraph)
        sub_node_index = subgraph.reindex_from_parrent_nodes(batch_nodes)
        
        batch_nodes = nodes
        graph_list.append((subgraph, batch_nodes, sub_node_index))
        
    graph_list = graph_list[::-1]
#     for k, v in graph._node_feat.items():
#         graph_list[0][0]._node_feat[k] = v
        
#     sub_node_index = subgraph.reindex_from_parrent_nodes(batch_nodes)

    return graph_list


class SampleDataGenerator(BaseDataGenerator): 
    def __init__(self,
                 graph_wrappers=None,
                 buf_size=1000,
                 batch_size=128,
                 num_workers=1,
                 sizes=[30, 30],
                 shuffle=True,
                 dataset=None,
                 nodes_idx=None):
        super(SampleDataGenerator, self).__init__(
            buf_size=buf_size,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=shuffle)
        self.sizes = sizes
        self.graph_wrappers = graph_wrappers
        self.dataset = dataset

        graph, labels = dataset[0]
        self.graph = graph
        self.num_nodes = graph.num_nodes
        if nodes_idx is not None:
            self.nodes_idx = nodes_idx
        else:
            self.nodes_idx = np.arange(self.num_nodes)
        
        self.labels_all = labels
        self.labels = labels[self.nodes_idx]
        
        self.sample_based_line_example(self.nodes_idx, self.labels)

    def sample_based_line_example(self, nodes_idx, labels): 
        self.line_examples = []
        Example = namedtuple('Example', ["node", "label"])
        for node, label in zip(nodes_idx, labels):
            self.line_examples.append(Example(node=node, label=label))
        print("Len Examples", len(self.line_examples))

    def batch_fn(self, batch_ex): 
        batch_nodes = []
        cc = 0
        batch_node_id = []
        batch_labels = []
        for ex in batch_ex:
            batch_nodes.append(ex.node)
            batch_labels.append(ex.label)

#         _graph_wrapper = copy.copy(self.graph_wrapper)
#         graph_list
        
        graph_list = k_hop_sampler(self.graph, self.sizes,
                                                 batch_nodes)   # -1 = 全采样操作
        
        feed_dict_all = {}
        
        for i in range(len(self.sizes)):
            feed_dict = self.graph_wrappers[i].to_feed(graph_list[i][0])
            feed_dict_all.update(feed_dict)
            if i == 0:
                feed_dict_all["batch_nodes_" + str(i)] = np.array(graph_list[i][1])
            feed_dict_all["sub_node_index_" + str(i)] = graph_list[i][2]
        
#         feed_dict = _graph_wrapper.to_feed(subgraph)
#         feed_dict["batch_nodes"] = np.array(batch_nodes)
#         feed_dict["sub_node_index"] = sub_node_index
        feed_dict_all["label_all"] = self.labels_all
        feed_dict_all["label"] = np.array(batch_labels, dtype="int64")
        return feed_dict_all