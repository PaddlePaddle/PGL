# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

# Cora Binary dataset
import os
import sys

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp

from .utils import save_graphs, load_graphs, save_info, load_info, makedirs, _get_dgl_url
from .utils import generate_mask_tensor
from .utils import deprecate_property, deprecate_function
from .dgl_dataset import DGLBuiltinDataset
from .. import convert
from .. import batch
from .. import backend as F
from ..convert import graph as dgl_graph
from ..convert import from_networkx, to_networkx

backend = os.environ.get('DGLBACKEND', 'pytorch')


def _pickle_load(pkl_file):
    if sys.version_info > (3, 0):
        return pkl.load(pkl_file, encoding='latin1')
    else:
        return pkl.load(pkl_file)


class CoraBinary(DGLBuiltinDataset):
    """A mini-dataset for binary classification task using Cora.
    After loaded, it has following members:
    graphs : list of :class:`~dgl.DGLGraph`
    pmpds : list of :class:`scipy.sparse.coo_matrix`
    labels : list of :class:`numpy.ndarray`
    Parameters
    -----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
        Whether to print out progress information. Default: True.
    """

    def __init__(self, raw_dir=None, force_reload=False, verbose=True):
        name = 'cora_binary'
        url = _get_dgl_url('dataset/cora_binary.zip')
        super(CoraBinary, self).__init__(
            name,
            url=url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose)

    def process(self):
        root = self.raw_path
        # load graphs
        self.graphs = []
        with open("{}/graphs.txt".format(root), 'r') as f:
            elist = []
            for line in f.readlines():
                if line.startswith('graph'):
                    if len(elist) != 0:
                        self.graphs.append(dgl_graph(tuple(zip(*elist))))
                    elist = []
                else:
                    u, v = line.strip().split(' ')
                    elist.append((int(u), int(v)))
            if len(elist) != 0:
                self.graphs.append(dgl_graph(tuple(zip(*elist))))
        with open("{}/pmpds.pkl".format(root), 'rb') as f:
            self.pmpds = _pickle_load(f)
        self.labels = []
        with open("{}/labels.txt".format(root), 'r') as f:
            cur = []
            for line in f.readlines():
                if line.startswith('graph'):
                    if len(cur) != 0:
                        self.labels.append(np.asarray(cur))
                    cur = []
                else:
                    cur.append(int(line.strip()))
            if len(cur) != 0:
                self.labels.append(np.asarray(cur))
        # sanity check
        assert len(self.graphs) == len(self.pmpds)
        assert len(self.graphs) == len(self.labels)

    def has_cache(self):
        graph_path = os.path.join(self.save_path, self.save_name + '.bin')
        if os.path.exists(graph_path):
            return True

        return False

    def save(self):
        """save the graph list and the labels"""
        graph_path = os.path.join(self.save_path, self.save_name + '.bin')
        labels = {}
        for i, label in enumerate(self.labels):
            labels['{}'.format(i)] = F.tensor(label)
        save_graphs(str(graph_path), self.graphs, labels)
        if self.verbose:
            print('Done saving data into cached files.')

    def load(self):
        graph_path = os.path.join(self.save_path, self.save_name + '.bin')
        self.graphs, labels = load_graphs(str(graph_path))

        self.labels = []
        for i in range(len(lables)):
            self.labels.append(labels['{}'.format(i)].asnumpy())
        # load pmpds under self.raw_path
        with open("{}/pmpds.pkl".format(self.raw_path), 'rb') as f:
            self.pmpds = _pickle_load(f)
        if self.verbose:
            print('Done loading data into cached files.')
        # sanity check
        assert len(self.graphs) == len(self.pmpds)
        assert len(self.graphs) == len(self.labels)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, i):
        r"""Gets the idx-th sample.
        Parameters
        -----------
        idx : int
            The sample index.
        Returns
        -------
        (dgl.DGLGraph, scipy.sparse.coo_matrix, int)
            The graph, scipy sparse coo_matrix and its label.
        """
        return (self.graphs[i], self.pmpds[i], self.labels[i])

    @property
    def save_name(self):
        return self.name + '_dgl_graph'

    @staticmethod
    def collate_fn(cur):
        graphs, pmpds, labels = zip(*cur)
        batched_graphs = batch.batch(graphs)
        batched_pmpds = sp.block_diag(pmpds)
        batched_labels = np.concatenate(labels, axis=0)
        return batched_graphs, batched_pmpds, batched_labels
