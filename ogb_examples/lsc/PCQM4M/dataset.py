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

import os
import sys
import json
import numpy as np
import glob
import copy
import time
import tqdm
import argparse
import pickle as pkl
from collections import OrderedDict, namedtuple

from ogb.lsc import PCQM4MDataset, PCQM4MEvaluator
from ogb.utils import smiles2graph

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import pgl
from pgl.utils.data.dataset import Dataset, StreamDataset, HadoopDataset
from pgl.utils.data import Dataloader
from pgl.utils.logger import log


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


class MolDataset(Dataset):
    def __init__(self, config, mode='train', transform=None):
        self.config = config
        self.mode = mode
        self.transform = transform
        self.raw_dataset = PCQM4MDataset(
            config.base_data_path, only_smiles=True)

        log.info("preprocess graph data in %s" % self.__class__.__name__)
        processed_path = os.path.join(self.raw_dataset.folder, "pgl_processed")
        if not os.path.exists(processed_path):
            os.makedirs(processed_path)
        data_file = os.path.join(processed_path, "graph_data.pkl")

        if os.path.exists(data_file):
            log.info("loading graph data from pkl file")
            self.graph_list = pkl.load(open(data_file, "rb"))
        else:
            log.info("loading graph data from smiles data")
            self.graph_list = []
            for i in tqdm.tqdm(range(len(self.raw_dataset))):
                # num_nodes, edge_index, node_feat, edge_feat, label
                smiles, label = self.raw_dataset[i]
                graph = smiles2graph(smiles)
                new_graph = {}
                new_graph["edges"] = graph["edge_index"].T
                new_graph["num_nodes"] = graph["num_nodes"]
                new_graph["node_feat"] = graph["node_feat"]
                new_graph["edge_feat"] = graph["edge_feat"]
                new_graph["label"] = label
                self.graph_list.append(new_graph)

            pkl.dump(self.graph_list, open(data_file, 'wb'))

    def get_idx_split(self):
        return self.raw_dataset.get_idx_split()

    def __getitem__(self, idx):
        if self.transform is not None:
            return self.transform(self.graph_list[idx])
        else:
            return self.graph_list[idx]

    def __len__(self):
        return len(self.graph_list)


class CollateFn(object):
    def __init__(self):
        pass

    def __call__(self, batch_data):
        graph_list = []
        labels = []
        for gdata in batch_data:
            g = pgl.Graph(
                edges=gdata['edges'],
                num_nodes=gdata['num_nodes'],
                node_feat={'feat': gdata['node_feat']},
                edge_feat={'feat': gdata['edge_feat']})
            graph_list.append(g)
            labels.append(gdata['label'])

        labels = np.array(labels, dtype="float32")
        g = pgl.Graph.batch(graph_list)

        return g, labels
