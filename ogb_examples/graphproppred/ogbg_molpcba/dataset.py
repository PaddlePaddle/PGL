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
import argparse
from collections import OrderedDict, namedtuple
from scipy.sparse import csr_matrix

from ogb.graphproppred import GraphPropPredDataset
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

import pgl
import paddle
from pgl.utils.data.dataset import Dataset, StreamDataset, HadoopDataset
from pgl.utils.data import Dataloader
from pgl.utils.logger import log

from utils.config import prepare_config, make_dir


def make_multihop_edges(gdata, k):
    gdata = copy.deepcopy(gdata)
    num_nodes = gdata['num_nodes']

    multihop_edges = []

    # distance 0: self loop means distance == 0
    hop0_edges = np.arange(0, num_nodes, dtype="int64").reshape(-1, 1)
    hop0_edges = np.repeat(hop0_edges, 2, axis=1)

    src = hop0_edges[:, 0]
    dst = hop0_edges[:, 1]
    value = np.zeros(shape=(len(src), ))
    A0 = csr_matrix((value, (src, dst)), shape=(num_nodes, num_nodes))

    # distance 1:
    hop1_edges = np.array(gdata['edges'])
    if len(hop1_edges.shape) != 2:
        return (None, None, None)

    src = hop1_edges[:, 0]
    dst = hop1_edges[:, 1]
    value = np.ones(shape=(len(src), )) * 2
    A1 = csr_matrix((value, (src, dst)), shape=(num_nodes, num_nodes))

    Ad = A1
    Ads = [A0, A1]
    for d in range(2, k + 1):
        Ad = csr_matrix.dot(Ad, A1) + A1
        coo_Ad = Ad.tocoo()
        row = coo_Ad.row
        col = coo_Ad.col
        value = np.ones(shape=(len(row), )) * (1 << d)
        Ad = csr_matrix((value, (row, col)), shape=(num_nodes, num_nodes))
        Ads.append(Ad)

    for d in range(len(Ads) - 1, 0, -1):
        # d and d-1
        A = Ads[d] + Ads[d - 1]
        A = A.tocoo()
        d_idx = ((A.data / (1 << d)) % 2 == 1) & (
            (A.data / (1 << (d - 1))) % 2 != 1)
        d_row = A.row[d_idx]
        d_col = A.col[d_idx]
        e = np.stack((d_row, d_col)).T
        multihop_edges.append(e)

    multihop_edges.append(
        hop0_edges)  # distance 0, not necessary to minimum, append directly.

    multihop_edges.reverse()
    multihop_edges[1] = hop1_edges
    gdata['mh_edges'] = multihop_edges
    main_graph = pgl.Graph(
        num_nodes=gdata['num_nodes'],
        edges=gdata['edges'],
        node_feat={'feat': gdata['nfeat']},
        edge_feat={'feat': gdata['efeat']})

    graph_list = []
    for d in range(k + 1):
        g = pgl.Graph(num_nodes=gdata['num_nodes'], edges=multihop_edges[d])
        graph_list.append(g)

    return (main_graph, graph_list, gdata['label'])


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
    def __init__(self, config, raw_dataset, mode='train', transform=None):
        self.config = config
        self.raw_dataset = raw_dataset
        self.mode = mode
        self.transform = transform

        log.info("preprocess graph data in %s" % self.__class__.__name__)
        self.graph_list = []
        for i in range(len(self.raw_dataset)):
            # num_nodes, edge_index, node_feat, edge_feat, label
            graph, label = self.raw_dataset[i]
            num_nodes = graph['num_nodes']
            node_feat = graph['node_feat'].copy()
            edges = list(zip(graph["edge_index"][0], graph["edge_index"][1]))
            edge_feat = graph['edge_feat'].copy()

            new_graph = {}
            new_graph['num_nodes'] = num_nodes
            new_graph['nfeat'] = node_feat
            new_graph['edges'] = edges
            new_graph['efeat'] = edge_feat
            new_graph['label'] = label

            self.graph_list.append(new_graph)

    def __getitem__(self, idx):
        if self.transform is not None:
            return self.transform(self.graph_list[idx], self.config.K)
        else:
            return self.graph_list[idx]

    def __len__(self):
        return len(self.graph_list)


class CollateFn(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, batch_data):
        graph_list = []
        multihop_graph_list = [[] for _ in range(self.config.K + 1)]
        labels = []
        for g, multihop_g_list, y in batch_data:
            if g is None:
                continue
            graph_list.append(g)
            labels.append(y)
            for d in range(self.config.K + 1):
                multihop_graph_list[d].append(multihop_g_list[d])

        labels = np.array(labels)
        batch_valid = (labels == labels).astype("bool")
        labels = np.nan_to_num(labels).astype("float32")

        g = pgl.Graph.batch(graph_list)
        multihop_graphs = []
        for g_list in multihop_graph_list:
            multihop_graphs.append(pgl.Graph.batch(g_list))

        return g, multihop_graphs, labels, batch_valid


if __name__ == "__main__":
    config = prepare_config("pcba_config.yaml", isCreate=False, isSave=False)
    raw_dataset = GraphPropPredDataset(name=config.dataset_name)
    ds = MolDataset(config, raw_dataset, transform=make_multihop_edges)
    splitted_index = raw_dataset.get_idx_split()
    train_ds = Subset(ds, splitted_index['train'], mode='train')
    valid_ds = Subset(ds, splitted_index['valid'], mode="valid")
    test_ds = Subset(ds, splitted_index['test'], mode="test")

    Fn = CollateFn(config)
    loader = Dataloader(
        train_ds, batch_size=3, shuffle=False, num_workers=4, collate_fn=Fn)
    for batch_data in loader:
        print("batch", batch_data[0][0].node_feat)
        g = pgl.Graph.batch(batch_data[0])
        print(g.node_feat)
        time.sleep(3)
