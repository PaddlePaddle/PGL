#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import sys
import os

from ogb.graphproppred import GraphPropPredDataset
import pgl
from pgl.utils.logger import log


class BaseDataset(object):
    def __init__(self):
        pass

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class Subset(BaseDataset):
    r"""
    Subset of a dataset at specified indices.
    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class Dataset(BaseDataset):
    def __init__(self, args):
        self.args = args
        self.raw_dataset = GraphPropPredDataset(name=args.dataset_name)
        self.num_tasks = self.raw_dataset.num_tasks
        self.eval_metrics = self.raw_dataset.eval_metric
        self.task_type = self.raw_dataset.task_type

        self.pgl_graph_list = []
        self.graph_label_list = []
        for i in range(len(self.raw_dataset)):
            graph, label = self.raw_dataset[i]
            edges = list(zip(graph["edge_index"][0], graph["edge_index"][1]))
            g = pgl.graph.Graph(num_nodes=graph["num_nodes"], edges=edges)

            if graph["edge_feat"] is not None:
                g.edge_feat["feat"] = graph["edge_feat"]

            if graph["node_feat"] is not None:
                g.node_feat["feat"] = graph["node_feat"]

            self.pgl_graph_list.append(g)
            self.graph_label_list.append(label)

    def __getitem__(self, idx):
        return self.pgl_graph_list[idx], self.graph_label_list[idx]

    def __len__(self):
        return len(slef.pgl_graph_list)

    def get_idx_split(self):
        return self.raw_dataset.get_idx_split()
