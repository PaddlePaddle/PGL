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
import torch

import os
import sys
import json
import numpy as np
import glob
import copy
import time
import argparse
from collections import OrderedDict, namedtuple
from scipy.sparse import csr_matrix
import pgl
import paddle
from pgl.utils.data.dataset import Dataset, StreamDataset, HadoopDataset
from pgl.utils.data import Dataloader
from pgl.utils.logger import log

from utils.config import prepare_config, make_dir
from ogb.graphproppred import GraphPropPredDataset
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims


class Subset(Dataset):
    r"""
    Subset of a dataset at specified indices.
    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices, mode='train'):
        self.dataset = dataset
        if paddle.distributed.get_world_size() == 1 or mode != "train":
            self.indices = indices
        else:
            self.indices = indices[int(paddle.distributed.get_rank())::int(
                paddle.distributed.get_world_size())]

        self.mode = mode

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class ShardedDataset(Dataset):
    """
    SharderDataset
    """

    def __init__(self, data, mode="train"):
        if paddle.distributed.get_world_size() == 1 or mode != "train":
            self.data = data
        else:
            self.data = data[int(paddle.distributed.get_rank())::int(
                paddle.distributed.get_world_size())]

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class MolDataset(Dataset):
    """
    Transfer raw ogb dataset to pgl dataset
    """

    def __init__(self, config, raw_dataset, mode='train', transform=None):
        self.config = config
        self.raw_dataset = raw_dataset
        self.mode = mode

        log.info("preprocess graph data in %s" % self.__class__.__name__)
        self.graph_list = []
        self.label = []
        for i in range(len(self.raw_dataset)):
            # num_nodes, edge_index, node_feat, edge_feat, label
            graph, label = self.raw_dataset[i]
            num_nodes = graph['num_nodes']
            node_feat = graph['node_feat'].copy()
            edges = list(zip(graph["edge_index"][0], graph["edge_index"][1]))
            edge_feat = graph['edge_feat'].copy()
            main_graph = pgl.Graph(
                num_nodes=num_nodes,
                edges=edges,
                node_feat={'feat': node_feat},
                edge_feat={'feat': edge_feat})
            self.graph_list.append(main_graph)
            self.label.append(label)

    def __getitem__(self, idx):
        return self.graph_list[idx], self.label[idx]

    def __len__(self):
        return len(self.graph_list)


class CollateFn(object):
    def __init__(self):
        pass

    def __call__(self, batch_data):
        graph_list = []
        labels = []
        for g, label in batch_data:
            if g is None:
                continue
            graph_list.append(g)
            labels.append(label)

        labels = np.array(labels)
        batch_valid = (labels == labels).astype("bool")
        labels = np.nan_to_num(labels).astype("float32")

        g = pgl.Graph.batch(graph_list)
        return g, labels, batch_valid
