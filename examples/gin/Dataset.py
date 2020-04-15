# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
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
This file implement the dataset for GIN model.
"""

import os
import sys
import numpy as np

from sklearn.model_selection import StratifiedKFold

import pgl
from pgl.utils.logger import log


def fold10_split(dataset, fold_idx=0, seed=0, shuffle=True):
    """10 fold splitter"""
    assert 0 <= fold_idx and fold_idx < 10, print(
        "fold_idx must be from 0 to 9.")

    skf = StratifiedKFold(n_splits=10, shuffle=shuffle, random_state=seed)
    labels = []
    for i in range(len(dataset)):
        g, c = dataset[i]
        labels.append(c)

    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, valid_idx = idx_list[fold_idx]

    log.info("train_set : test_set == %d : %d" %
             (len(train_idx), len(valid_idx)))
    return Subset(dataset, train_idx), Subset(dataset, valid_idx)


def random_split(dataset, split_ratio=0.7, seed=0, shuffle=True):
    """random splitter"""
    np.random.seed(seed)
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    split = int(split_ratio * len(dataset))
    train_idx, valid_idx = indices[:split], indices[split:]

    log.info("train_set : test_set == %d : %d" %
             (len(train_idx), len(valid_idx)))
    return Subset(dataset, train_idx), Subset(dataset, valid_idx)


class BaseDataset(object):
    """BaseDataset"""

    def __init__(self):
        pass

    def __getitem__(self, idx):
        """getitem"""
        raise NotImplementedError

    def __len__(self):
        """len"""
        raise NotImplementedError


class Subset(BaseDataset):
    """
    Subset of a dataset at specified indices.
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        """getitem"""
        return self.dataset[self.indices[idx]]

    def __len__(self):
        """len"""
        return len(self.indices)


class GINDataset(BaseDataset):
    """Dataset for Graph Isomorphism Network (GIN)
    Adapted from https://github.com/weihua916/powerful-gnns/blob/master/dataset.zip.
    """

    def __init__(self,
                 data_path,
                 dataset_name,
                 self_loop,
                 degree_as_nlabel=False):
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.self_loop = self_loop
        self.degree_as_nlabel = degree_as_nlabel

        self.graph_list = []
        self.glabel_list = []

        # relabel
        self.glabel_dict = {}
        self.nlabel_dict = {}
        self.elabel_dict = {}
        self.ndegree_dict = {}

        # global num
        self.num_graph = 0  # total graphs number
        self.n = 0  # total nodes number
        self.m = 0  # total edges number

        # global num of classes
        self.gclasses = 0
        self.nclasses = 0
        self.eclasses = 0
        self.dim_nfeats = 0

        # flags
        self.degree_as_nlabel = degree_as_nlabel
        self.nattrs_flag = False
        self.nlabels_flag = False

        self._load_data()

    def __len__(self):
        """return the number of graphs"""
        return len(self.graph_list)

    def __getitem__(self, idx):
        """getitem"""
        return self.graph_list[idx], self.glabel_list[idx]

    def _load_data(self):
        """Loads dataset
        """
        filename = os.path.join(self.data_path, self.dataset_name,
                                "%s.txt" % self.dataset_name)
        log.info("loading data from %s" % filename)

        with open(filename, 'r') as reader:
            # first line --> N, means total number of graphs
            self.num_graph = int(reader.readline().strip())

            for i in range(self.num_graph):
                if (i + 1) % int(self.num_graph / 10) == 0:
                    log.info("processing graph %s" % (i + 1))
                graph = dict()
                # second line --> [num_node, label] 
                # means [node number of a graph, class label of a graph]
                grow = reader.readline().strip().split()
                n_nodes, glabel = [int(w) for w in grow]

                # relabel graphs
                if glabel not in self.glabel_dict:
                    mapped = len(self.glabel_dict)
                    self.glabel_dict[glabel] = mapped

                graph['num_nodes'] = n_nodes
                self.glabel_list.append(self.glabel_dict[glabel])

                nlabels = []
                node_features = []
                num_edges = 0
                edges = []

                for j in range(graph['num_nodes']):
                    slots = reader.readline().strip().split()

                    # handle edges and node feature(if has)
                    tmp = int(slots[
                        1]) + 2  # tmp == 2 + num_edges of current node
                    if tmp == len(slots):
                        # no node feature
                        nrow = [int(w) for w in slots]
                        nfeat = None
                    elif tmp < len(slots):
                        nrow = [int(w) for w in slots[:tmp]]
                        nfeat = [float(w) for w in slots[tmp:]]
                        node_features.append(nfeat)
                    else:
                        raise Exception('edge number is not correct!')

                    # relabel nodes if is has labels
                    # if it doesn't have node labels, then every nrow[0] == 0
                    if not nrow[0] in self.nlabel_dict:
                        mapped = len(self.nlabel_dict)
                        self.nlabel_dict[nrow[0]] = mapped

                    nlabels.append(self.nlabel_dict[nrow[0]])
                    num_edges += nrow[1]
                    edges.extend([(j, u) for u in nrow[2:]])

                    if self.self_loop:
                        num_edges += 1
                        edges.append((j, j))

                if node_features != []:
                    node_features = np.stack(node_features)
                    graph['attr'] = node_features
                    self.nattrs_flag = True
                else:
                    node_features = None
                    graph['attr'] = node_features

                graph['nlabel'] = np.array(
                    nlabels, dtype="int64").reshape(-1, 1)
                if len(self.nlabel_dict) > 1:
                    self.nlabels_flag = True

                graph['edges'] = edges
                assert num_edges == len(edges)

                g = pgl.graph.Graph(
                    num_nodes=graph['num_nodes'],
                    edges=graph['edges'],
                    node_feat={
                        'nlabel': graph['nlabel'],
                        'attr': graph['attr']
                    })

                self.graph_list.append(g)

                # update statistics of graphs
                self.n += graph['num_nodes']
                self.m += num_edges

        # if no attr
        if not self.nattrs_flag:
            log.info('there are no node features in this dataset!')
            label2idx = {}
            # generate node attr by node degree
            if self.degree_as_nlabel:
                log.info('generate node features by node degree...')
                nlabel_set = set([])
                for g in self.graph_list:

                    g.node_feat['nlabel'] = g.indegree()
                    # extracting unique node labels
                    nlabel_set = nlabel_set.union(set(g.node_feat['nlabel']))
                    g.node_feat['nlabel'] = g.node_feat['nlabel'].reshape(-1,
                                                                          1)

                nlabel_set = list(nlabel_set)
                # in case the labels/degrees are not continuous number
                self.ndegree_dict = {
                    nlabel_set[i]: i
                    for i in range(len(nlabel_set))
                }
                label2idx = self.ndegree_dict
            # generate node attr by node label
            else:
                log.info('generate node features by node label...')
                label2idx = self.nlabel_dict

            for g in self.graph_list:
                attr = np.zeros((g.num_nodes, len(label2idx)))
                idx = [
                    label2idx[tag]
                    for tag in g.node_feat['nlabel'].reshape(-1, )
                ]
                attr[:, idx] = 1
                g.node_feat['attr'] = attr.astype("float32")

        # after load, get the #classes and #dim
        self.gclasses = len(self.glabel_dict)
        self.nclasses = len(self.nlabel_dict)
        self.eclasses = len(self.elabel_dict)
        self.dim_nfeats = len(self.graph_list[0].node_feat['attr'][0])

        message = "finished loading data\n"
        message += """
                    num_graph: %d
                    num_graph_class: %d
                    total_num_nodes: %d
                    node Classes: %d
                    node_features_dim: %d
                    num_edges: %d
                    edge_classes: %d
                    Avg. of #Nodes: %.2f
                    Avg. of #Edges: %.2f
                    Graph Relabeled: %s
                    Node Relabeled: %s
                    Degree Relabeled(If degree_as_nlabel=True): %s""" % (
            self.num_graph,
            self.gclasses,
            self.n,
            self.nclasses,
            self.dim_nfeats,
            self.m,
            self.eclasses,
            self.n / self.num_graph,
            self.m / self.num_graph,
            self.glabel_dict,
            self.nlabel_dict,
            self.ndegree_dict, )
        log.info(message)


if __name__ == "__main__":
    gindataset = GINDataset(
        "./dataset/", "MUTAG", self_loop=True, degree_as_nlabel=False)
