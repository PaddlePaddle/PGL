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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

from dataloader.base_dataloader import BaseDataGenerator
from utils.to_undirected import to_undirected
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from pgl.contrib.ogb.nodeproppred.dataset_pgl import PglNodePropPredDataset
from pgl.sample import graph_saint_random_walk_sample
from ogb.nodeproppred import Evaluator
import tqdm
from collections import namedtuple
import pgl
import numpy as np
import copy


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
    # for batch_train_samples, batch_train_labels in batch_info:
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
        nodes = flat_node_and_edge(nodes)
        # Find new nodes
        start_nodes = list(set(nodes) - set(last_nodes))
        if len(start_nodes) == 0:
            break

    subgraph = graph.subgraph(
        nodes=nodes, edges=edges, with_node_feat=True, with_edge_feat=True)
    sub_node_index = subgraph.reindex_from_parrent_nodes(batch_nodes)

    return subgraph, sub_node_index


def graph_saint_randomwalk_sampler(graph, batch_nodes, max_depth=3):
    subgraph = graph_saint_random_walk_sample(graph, batch_nodes, max_depth)
    sub_node_index = subgraph.reindex_from_parrent_nodes(batch_nodes)
    return subgraph, sub_node_index


class ArxivDataGenerator(BaseDataGenerator):
    def __init__(self,
                 graph_wrapper=None,
                 buf_size=1000,
                 batch_size=128,
                 num_workers=1,
                 samples=[30, 30],
                 shuffle=True,
                 phase="train"):
        super(ArxivDataGenerator, self).__init__(
            buf_size=buf_size,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=shuffle)
        self.samples = samples
        self.d_name = "ogbn-arxiv"
        self.graph_wrapper = graph_wrapper
        dataset = PglNodePropPredDataset(name=self.d_name)
        splitted_idx = dataset.get_idx_split()
        self.phase = phase
        graph, label = dataset[0]
        graph = to_undirected(graph)
        self.graph = graph
        self.num_nodes = graph.num_nodes
        if self.phase == 'train':
            nodes_idx = splitted_idx["train"]
            labels = label[nodes_idx]
        elif self.phase == "valid":
            nodes_idx = splitted_idx["valid"]
            labels = label[nodes_idx]
        elif self.phase == "test":
            nodes_idx = splitted_idx["test"]
            labels = label[nodes_idx]
        self.nodes_idx = nodes_idx
        self.labels = labels
        self.sample_based_line_example(nodes_idx, labels)

    def sample_based_line_example(self, nodes_idx, labels):
        self.line_examples = []
        Example = namedtuple('Example', ["node", "label"])
        for node, label in zip(nodes_idx, labels):
            self.line_examples.append(Example(node=node, label=label))
        print("Phase", self.phase)
        print("Len Examples", len(self.line_examples))

    def batch_fn(self, batch_ex):
        batch_nodes = []
        cc = 0
        batch_node_id = []
        batch_labels = []
        for ex in batch_ex:
            batch_nodes.append(ex.node)
            batch_labels.append(ex.label)

        _graph_wrapper = copy.copy(self.graph_wrapper)
        #if self.phase == "train":
        #    subgraph, sub_node_index = graph_saint_randomwalk_sampler(self.graph, batch_nodes)
        #else:
        subgraph, sub_node_index = k_hop_sampler(self.graph, self.samples,
                                                 batch_nodes)

        feed_dict = _graph_wrapper.to_feed(subgraph)
        feed_dict["batch_nodes"] = sub_node_index
        feed_dict["labels"] = np.array(batch_labels, dtype="int64")
        return feed_dict
