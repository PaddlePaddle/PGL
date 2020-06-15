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

import sys
import os
import random
import pgl
from pgl.utils.logger import log
from pgl.graph import Graph, MultiGraph
import numpy as np
import pickle

class BaseDataset(object):
    def __init__(self):
        pass

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class Subset(BaseDataset):
    """Subset of a dataset at specified indices.
    
    Args:
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
        
        with open('data/%s.pkl' % args.dataset_name, 'rb') as f:
            graphs_info_list = pickle.load(f)

        self.pgl_graph_list = []
        self.graph_label_list = []
        for i in range(len(graphs_info_list) - 1):
            graph = graphs_info_list[i]
            edges_l, edges_r = graph["edge_src"], graph["edge_dst"]
            
            # add self-loops
            if self.args.dataset_name != "FRANKENSTEIN":
                num_nodes = graph["num_nodes"]
                x = np.arange(0, num_nodes)
                edges_l = np.append(edges_l, x)
                edges_r = np.append(edges_r, x)
            
            edges = list(zip(edges_l, edges_r))
            g = pgl.graph.Graph(num_nodes=graph["num_nodes"], edges=edges)
            g.node_feat["feat"] = graph["node_feat"]
            self.pgl_graph_list.append(g)
            self.graph_label_list.append(graph["label"])
            
        self.num_classes = graphs_info_list[-1]["num_classes"]
        self.num_features = graphs_info_list[-1]["num_features"]

    def __getitem__(self, idx):
        return self.pgl_graph_list[idx], self.graph_label_list[idx]

    def shuffle(self):
        """shuffle the dataset.
        """
        cc = list(zip(self.pgl_graph_list, self.graph_label_list))
        random.seed(self.args.seed)
        random.shuffle(cc)
        a, b = zip(*cc)
        self.pgl_graph_list[:], self.graph_label_list[:] = a, b

    def __len__(self):
        return len(self.pgl_graph_list)
