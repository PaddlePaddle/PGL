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
    This package implements some benchmark dataset for graph network
    and node representation learning.
"""

import os
import io
import sys

import numpy as np
import pickle as pkl

from pgl.graph import Graph

__all__ = [
    "CitationDataset",
    "CoraDataset",
    "ArXivDataset",
    "BlogCatalogDataset",
    "RedditDataset",
    "OgbnArxivDataset",
]


def get_default_data_dir(name):
    """Get data path name"""
    dir_path = os.path.abspath(os.path.dirname(__file__))
    dir_path = os.path.join(dir_path, 'data')
    filepath = os.path.join(dir_path, name)
    return filepath


def _pickle_load(pkl_file):
    """Load pickle"""
    if sys.version_info > (3, 0):
        return pkl.load(pkl_file, encoding='latin1')
    else:
        return pkl.load(pkl_file)


def _parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


class CitationDataset(object):
    """Citation dataset helps to create data for citation dataset (Pubmed and Citeseer).

    Args:

        name (str): The name for the dataset ("pubmed" or "citeseer").

        symmetry_edges (bool): Whether to create symmetry edges.

        self_loop (bool):  Whether to contain self loop edges.

    Attributes:

        graph (pgl.Graph): The :code:`Graph` data object.

        y (numpy.ndarray): Labels for each nodes.

        num_classes (int): Number of classes.

        train_index (numpy.ndarray): The index for nodes in training set.

        val_index (numpy.ndarray): The index for nodes in validation set.

        test_index (numpy.ndarray): The index for nodes in test set.

    """

    def __init__(self, name, symmetry_edges=True, self_loop=True):
        self.path = get_default_data_dir(name)
        self.symmetry_edges = symmetry_edges
        self.self_loop = self_loop
        self.name = name
        self._load_data()

    def _load_data(self):
        """Load data
        """
        import networkx as nx
        objnames = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(objnames)):
            with open("{}/ind.{}.{}".format(self.path, self.name, objnames[i]),
                      'rb') as f:
                objects.append(_pickle_load(f))

        x, y, tx, ty, allx, ally, _graph = objects
        test_idx_reorder = _parse_index_file("{}/ind.{}.test.index".format(
            self.path, self.name))
        test_idx_range = np.sort(test_idx_reorder)

        allx = allx.todense()
        tx = tx.todense()
        if self.name == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(
                min(test_idx_reorder), max(test_idx_reorder) + 1)
            tx_extended = np.zeros(
                (len(test_idx_range_full), x.shape[1]), dtype="float32")
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros(
                (len(test_idx_range_full), y.shape[1]), dtype="float32")
            ty_extended[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_extended

        features = np.vstack([allx, tx])
        features[test_idx_reorder, :] = features[test_idx_range, :]
        features = features / (np.sum(features, axis=-1) + 1e-15)
        features = np.array(features, dtype="float32")
        _graph = nx.DiGraph(nx.from_dict_of_lists(_graph))

        onehot_labels = np.vstack((ally, ty))
        onehot_labels[test_idx_reorder, :] = onehot_labels[test_idx_range, :]
        labels = np.argmax(onehot_labels, 1)

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)
        all_edges = []
        for i in _graph.edges():
            u, v = tuple(i)
            all_edges.append((u, v))
            if self.symmetry_edges:
                all_edges.append((v, u))

        if self.self_loop:
            for i in range(_graph.number_of_nodes()):
                all_edges.append((i, i))
        all_edges = list(set(all_edges))

        self.graph = Graph(
            num_nodes=_graph.number_of_nodes(),
            edges=all_edges,
            node_feat={"words": features})
        self.y = np.array(labels, dtype="int64")
        self.num_classes = onehot_labels.shape[1]
        self.train_index = np.array(idx_train, dtype="int32")
        self.val_index = np.array(idx_val, dtype="int32")
        self.test_index = np.array(idx_test, dtype="int32")


class CoraDataset(object):
    """Cora dataset implementation.

    Args:

        symmetry_edges (bool): Whether to create symmetry edges.

        self_loop (bool):  Whether to contain self loop edges.

    Attributes:

        graph (pgl.Graph): The :code:`Graph` data object.

        y (numpy.ndarray): Labels for each nodes.

        num_classes (int): Number of classes.

        train_index (numpy.ndarray): The index for nodes in training set.

        val_index (numpy.ndarray): The index for nodes in validation set.

        test_index (numpy.ndarray): The index for nodes in test set.

    """

    def __init__(self, symmetry_edges=True, self_loop=True):
        self.path = get_default_data_dir("cora")
        self.symmetry_edges = symmetry_edges
        self.self_loop = self_loop
        self._load_data()

    def _load_data(self):
        """Load data"""
        content = os.path.join(self.path, 'cora.content')
        cite = os.path.join(self.path, 'cora.cites')
        node_feature = []
        paper_ids = []
        y = []
        y_dict = {}
        with open(content, 'r') as f:
            for line in f:
                line = line.strip().split()
                paper_id = int(line[0])
                paper_class = line[-1]
                if paper_class not in y_dict:
                    y_dict[paper_class] = len(y_dict)
                feature = [int(i) for i in line[1:-1]]
                feature_array = np.array(feature, dtype="float32")
                # Normalize
                feature_array = feature_array / (np.sum(feature_array) + 1e-15)
                node_feature.append(feature_array)
                y.append(y_dict[paper_class])
                paper_ids.append(paper_id)
        paper2vid = dict([(v, k) for (k, v) in enumerate(paper_ids)])
        num_nodes = len(paper_ids)
        node_feature = np.array(node_feature, dtype="float32")

        all_edges = []
        with open(cite, 'r') as f:
            for line in f:
                u, v = line.split()
                u = paper2vid[int(u)]
                v = paper2vid[int(v)]
                all_edges.append((u, v))
                if self.symmetry_edges:
                    all_edges.append((v, u))

        if self.self_loop:
            for i in range(num_nodes):
                all_edges.append((i, i))

        all_edges = list(set(all_edges))
        self.graph = Graph(
            num_nodes=num_nodes,
            edges=all_edges,
            node_feat={"words": node_feature})
        perm = np.arange(0, num_nodes)
        #np.random.shuffle(perm)
        self.train_index = perm[:140]
        self.val_index = perm[200:500]
        self.test_index = perm[500:1500]
        self.y = np.array(y, dtype="int64")
        self.num_classes = len(y_dict)


class BlogCatalogDataset(object):
    """BlogCatalog dataset implementation.

    Args:

        symmetry_edges (bool): Whether to create symmetry edges.

        self_loop (bool):  Whether to contain self loop edges.

    Attributes:

        graph (pgl.Graph): The :code:`Graph` data object.

        num_groups (int): Number of classes.

        train_index (numpy.ndarray): The index for nodes in training set.

        test_index (numpy.ndarray): The index for nodes in validation set.

    """

    def __init__(self, symmetry_edges=True, self_loop=False):
        self.path = get_default_data_dir("BlogCatalog")
        self.num_groups = 39
        self.symmetry_edges = symmetry_edges
        self.self_loop = self_loop
        self._load_data()

    def _load_data(self):
        edge_path = os.path.join(self.path, 'edges.csv')
        node_path = os.path.join(self.path, 'nodes.csv')
        group_edge_path = os.path.join(self.path, 'group-edges.csv')

        all_edges = []

        with io.open(node_path) as inf:
            num_nodes = len(inf.readlines())

        node_feature = np.zeros((num_nodes, self.num_groups))

        with io.open(group_edge_path) as inf:
            for line in inf:
                node_id, group_id = line.strip('\n').split(',')
                node_id, group_id = int(node_id) - 1, int(group_id) - 1
                node_feature[node_id][group_id] = 1

        with io.open(edge_path) as inf:
            for line in inf:
                u, v = line.strip('\n').split(',')
                u, v = int(u) - 1, int(v) - 1
                all_edges.append((u, v))
                if self.symmetry_edges:
                    all_edges.append((v, u))

        if self.self_loop:
            for i in range(num_nodes):
                all_edges.append((i, i))

        all_edges = list(set(all_edges))
        self.graph = Graph(
            num_nodes=num_nodes,
            edges=all_edges,
            node_feat={"group_id": node_feature})

        perm = np.arange(0, num_nodes)
        np.random.shuffle(perm)
        train_num = int(num_nodes * 0.5)
        self.train_index = perm[:train_num]
        self.test_index = perm[train_num:]


class ArXivDataset(object):
    """ArXiv dataset implementation.

    Args:

        np_random_seed (int): The random seed for numpy.

    Attributes:

        graph (pgl.Graph): The :code:`Graph` data object.

    """

    def __init__(self, np_random_seed=123):
        self.path = get_default_data_dir("arXiv")
        self.np_random_seed = np_random_seed
        self._load_data()

    def _load_data(self):
        np.random.seed(self.np_random_seed)
        edge_path = os.path.join(self.path, 'ca-AstroPh.txt')

        bi_edges = set()
        self.neg_edges = []
        self.pos_edges = []
        self.node2id = dict()

        def node_id(node):
            if node not in self.node2id:
                self.node2id[node] = len(self.node2id)
            return self.node2id[node]

        with io.open(edge_path) as inf:
            for _ in range(4):
                inf.readline()
            for line in inf:
                u, v = line.strip('\n').split('\t')
                u, v = node_id(u), node_id(v)
                if u < v:
                    bi_edges.add((u, v))
                else:
                    bi_edges.add((v, u))

        num_nodes = len(self.node2id)

        while len(self.neg_edges) < len(bi_edges) // 2:
            random_edges = np.random.choice(num_nodes, [len(bi_edges), 2])
            for (u, v) in random_edges:
                if u != v and (u, v) not in bi_edges and (v, u
                                                          ) not in bi_edges:
                    self.neg_edges.append((u, v))
                    if len(self.neg_edges) == len(bi_edges) // 2:
                        break

        bi_edges = list(bi_edges)
        np.random.shuffle(bi_edges)
        self.pos_edges = bi_edges[:len(bi_edges) // 2]
        bi_edges = bi_edges[len(bi_edges) // 2:]
        all_edges = []
        for edge in bi_edges:
            u, v = edge
            all_edges.append((u, v))
            all_edges.append((v, u))
        self.graph = Graph(num_nodes=num_nodes, edges=all_edges)


class RedditDataset(object):
    """Reddit dataset implementation.

    Args:

        normalize (bool): Whether to normalize feature.

        symmetry (bool): Whether to create symmetry edges.

    Attributes:

        graph (pgl.Graph): The :code:`Graph` data object.

        feature (numpy.ndarray): The feature of nodes.

        num_classes (int): Number of classes.

        train_index (numpy.ndarray): The index for nodes in training set.

        val_index (numpy.ndarray): The index for nodes in validation set.

        test_index (numpy.ndarray): The index for nodes in test set.

        train_label (numpy.ndarray): The label for nodes in training set.

        val_label (numpy.ndarray): The label for nodes in validation set.

        test_label (numpy.ndarray): The label for nodes in test set.

    """

    def __init__(self, normalize=True, symmetry=True):
        download_help_str = r"""
            data from https://github.com/matenure/FastGCN/issues/8

            You can download from Aistudio:
                https://aistudio.baidu.com/aistudio/datasetdetail/39616
            or google drive:
            reddit_adj.npz: https://drive.google.com/open?id=174vb0Ws7Vxk_QTUtxqTgDHSQ4El4qDHt
            reddit.npz: https://drive.google.com/open?id=19SphVl_Oe8SJ1r87Hr5a6znx3nJu1F2J
            """
        self.path = get_default_data_dir("reddit")
        if not os.path.exists(self.path):
            raise ValueError(
                "\n Please download the dataset to \n "
                " \t{} \n before use it.\n More information about download: {}".
                format(self.path, download_help_str))
        self._load_data(normalize, symmetry)

    def _load_data(self, normalize=True, symmetry=True):
        from sklearn.preprocessing import StandardScaler
        import scipy.sparse as sp

        data = np.load(os.path.join(self.path, "reddit.npz"))
        adj = sp.load_npz(os.path.join(self.path, "reddit_adj.npz"))
        if symmetry:
            adj = adj + adj.T
        adj = adj.tocoo()
        src = adj.row
        dst = adj.col

        num_classes = 41
        train_label = data['y_train']
        val_label = data['y_val']
        test_label = data['y_test']

        train_index = data['train_index']
        val_index = data['val_index']
        test_index = data['test_index']

        feature = data["feats"].astype("float32")

        if normalize:
            scaler = StandardScaler()
            scaler.fit(feature[train_index])
            feature = scaler.transform(feature)

        graph = Graph(num_nodes=feature.shape[0], edges=list(zip(src, dst)))

        self.graph = graph
        self.train_index = train_index
        self.train_label = train_label
        self.val_label = val_label
        self.val_index = val_index
        self.test_index = test_index
        self.test_label = test_label
        self.feature = feature
        self.num_classes = 41


class OgbnArxivDataset(object):
    """Ogbn Arxiv Dataset import and implementation.

    Attributes:

        graph (pgl.Graph): The :code:`Graph` data object.

        feature (numpy.ndarray): The feature of all nodes.

        y (numpy.ndarray): Labels for each nodes.

        num_classes (int): Number of classes.

        train_index (numpy.ndarray): The index for nodes in training set.

        val_index (numpy.ndarray): The index for nodes in validation set.

        test_index (numpy.ndarray): The index for nodes in test set.

    """

    def __init__(self):
        try:
            from ogb.nodeproppred import NodePropPredDataset
        except:
            raise ImportError(
                "Please run `pip install ogb` to install ogb library.")

        self.dataset = NodePropPredDataset(name="ogbn-arxiv")
        self._load_data()

    def _load_data(self):
        split_idx = self.dataset.get_idx_split()
        train_idx = split_idx["train"]
        valid_idx = split_idx["valid"]
        test_idx = split_idx["test"]
        ogb_graph, label = self.dataset[0]

        edges = ogb_graph["edge_index"].T
        graph = Graph(num_nodes=ogb_graph["num_nodes"], edges=edges)

        self.graph = graph
        self.feature = ogb_graph["node_feat"]
        self.y = label
        self.num_classes = self.dataset.num_classes
        self.train_index = train_idx
        self.val_index = valid_idx
        self.test_index = test_idx
