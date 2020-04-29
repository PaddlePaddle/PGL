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
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from ogb.linkproppred import LinkPropPredDataset
from ogb.linkproppred import Evaluator
import tqdm
from collections import namedtuple
import pgl
import numpy as np


class PPADataGenerator(BaseDataGenerator):
    def __init__(self,
                 graph_wrapper=None,
                 buf_size=1000,
                 batch_size=128,
                 num_workers=1,
                 shuffle=True,
                 phase="train"):
        super(PPADataGenerator, self).__init__(
            buf_size=buf_size,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=shuffle)

        self.d_name = "ogbl-ppa"
        self.graph_wrapper = graph_wrapper
        dataset = LinkPropPredDataset(name=self.d_name)
        splitted_edge = dataset.get_edge_split()
        self.phase = phase
        graph = dataset[0]
        edges = graph["edge_index"].T
        #self.graph = pgl.graph.Graph(num_nodes=graph["num_nodes"],
        #       edges=edges, 
        #       node_feat={"nfeat": graph["node_feat"],
        #             "node_id": np.arange(0, graph["num_nodes"], dtype="int64").reshape(-1, 1) })

        #self.graph.indegree()
        self.num_nodes = graph["num_nodes"]
        if self.phase == 'train':
            edges = splitted_edge["train"]["edge"]
            labels = np.ones(len(edges))
        elif self.phase == "valid":
            # Compute the embedding for all the nodes
            pos_edges = splitted_edge["valid"]["edge"]
            neg_edges = splitted_edge["valid"]["edge_neg"]
            pos_labels = np.ones(len(pos_edges))
            neg_labels = np.zeros(len(neg_edges))
            edges = np.vstack([pos_edges, neg_edges])
            labels = pos_labels.tolist() + neg_labels.tolist()
        elif self.phase == "test":
            # Compute the embedding for all the nodes
            pos_edges = splitted_edge["test"]["edge"]
            neg_edges = splitted_edge["test"]["edge_neg"]
            pos_labels = np.ones(len(pos_edges))
            neg_labels = np.zeros(len(neg_edges))
            edges = np.vstack([pos_edges, neg_edges])
            labels = pos_labels.tolist() + neg_labels.tolist()

        self.line_examples = []
        Example = namedtuple('Example', ['src', "dst", "label"])
        for edge, label in zip(edges, labels):
            self.line_examples.append(
                Example(
                    src=edge[0], dst=edge[1], label=label))
        print("Phase", self.phase)
        print("Len Examples", len(self.line_examples))

    def batch_fn(self, batch_ex):
        batch_src = []
        batch_dst = []
        join_graph = []
        cc = 0
        batch_node_id = []
        batch_labels = []
        for ex in batch_ex:
            batch_src.append(ex.src)
            batch_dst.append(ex.dst)
            batch_labels.append(ex.label)

        if self.phase == "train":
            for num in range(1):
                rand_src = np.random.randint(
                    low=0, high=self.num_nodes, size=len(batch_ex))
                rand_dst = np.random.randint(
                    low=0, high=self.num_nodes, size=len(batch_ex))
                batch_src = batch_src + rand_src.tolist()
                batch_dst = batch_dst + rand_dst.tolist()
                batch_labels = batch_labels + np.zeros_like(
                    rand_src, dtype="int64").tolist()

        feed_dict = {}

        feed_dict["batch_src"] = np.array(batch_src, dtype="int64")
        feed_dict["batch_dst"] = np.array(batch_dst, dtype="int64")
        feed_dict["labels"] = np.array(batch_labels, dtype="int64")
        return feed_dict
