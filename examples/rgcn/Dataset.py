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
This package implements some benchmark dataset for graph network
and node representation learning.
"""
import os
import glob
import numpy as np
import pandas as pd
import pgl
from pgl.utils.logger import log

class Dataset(object):
    """Cora dataset implementation
    Args:
        symmetry_edges: Whether to create symmetry edges.
        self_loop:  Whether to contain self loop edges.

    Attributes:
        graph: The :code:`Graph` data object
        y: Labels for each nodes
        num_classes: Number of classes.
        train_index: The index for nodes in training set.
        val_index: The index for nodes in validation set.
        test_index: The index for nodes in test set.
    """

    def __init__(self, args, symmetry_edges=True, self_loop=True):
        self.model_path = args.init_checkpoint
        self.path = args.dataset
        self.symmetry_edges = symmetry_edges
        self.self_loop = self_loop
        self.load_data()

    def build_heter_graph(self, data_path, num_nodes):
        """
        build_heter_graph
        """
        edges = {}
        idx = 0
        for filename in glob.glob(os.path.join(data_path, '*')):
            try:
                e = pd.read_csv(filename, header=None, sep="\t").values
                edges['etype%s' % idx] = e
                idx += 1
            except Exception as e:
                # log.info(e)
                continue

        node_types = [(i, "n") for i in range(num_nodes)]

        graph = pgl.heter_graph.HeterGraph(
            num_nodes=num_nodes,
            edges=edges,
            node_types=node_types)
            # node_feat=node_features)
        
        return graph

    def load_data(self):
        """Load data"""
        if os.path.exists(os.path.join(self.model_path, "data")):
            self.path = os.path.join(self.model_path, "data")

        nodes_file = os.path.join(self.path, 'node_labels.txt')
        train_file = os.path.join(self.path, 'train_idx.txt')
        test_file = os.path.join(self.path, 'test_idx.txt')
        edges_file = os.path.join(self.path, 'edges')

        node_labels = pd.read_csv(nodes_file, header=None, sep="\t").values.astype("int64")
        self.node_labels = node_labels[:, 1:2]
        num_nodes = len(node_labels)
        self.train_index = pd.read_csv(train_file, header=None, sep="\t").values.astype("int64").reshape(-1, ).tolist()
        self.val_index = pd.read_csv(test_file, header=None, sep="\t").values.astype("int64").reshape(-1, ).tolist()
        self.test_index = pd.read_csv(test_file, header=None, sep="\t").values.astype("int64").reshape(-1, ).tolist()

        self.graph = self.build_heter_graph(edges_file, num_nodes)
            
        log.info('total %d nodes are loaded' % (self.graph.num_nodes))
        log.info('total %d edges are loaded' % (len(self.graph.edge_types_info())))
        log.info('nodes feat_info: {}' .format(self.graph.node_feat_info()))
