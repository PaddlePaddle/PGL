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
This file loads and preprocesses the dataset for GATNE model.
"""

import sys
import os
import tqdm
import numpy as np
import logging
import random
from pgl import heter_graph
import pickle as pkl


class Dataset(object):
    """Implementation of Dataset class

    This is a simple implementation of loading and processing dataset for GATNE model.

    Args:
        config: dict, some configure parameters.
    """

    def __init__(self, config):
        self.train_edges_file = config['data_path'] + 'train.txt'
        self.valid_edges_file = config['data_path'] + 'valid.txt'
        self.test_edges_file = config['data_path'] + 'test.txt'
        self.nodes_file = config['data_path'] + 'nodes.txt'

        self.config = config

        self.word2index = self.load_word2index()

        self.build_graph()

        self.valid_data = self.load_test_data(self.valid_edges_file)
        self.test_data = self.load_test_data(self.test_edges_file)

    def build_graph(self):
        """Build pgl heterogeneous graph. 
        """
        edge_data_by_type, all_edges, all_nodes = self.load_training_data(
            self.train_edges_file,
            slf_loop=self.config['slf_loop'],
            symmetry_edge=self.config['symmetry_edge'])

        num_nodes = len(all_nodes)
        node_features = {
            'index': np.array(
                [i for i in range(num_nodes)], dtype=np.int64).reshape(-1, 1)
        }

        self.graph = heter_graph.HeterGraph(
            num_nodes=num_nodes,
            edges=edge_data_by_type,
            node_types=None,
            node_feat=node_features)

        self.edge_types = sorted(self.graph.edge_types_info())
        logging.info('total %d nodes are loaded' % (self.graph.num_nodes))

    def load_training_data(self, file_, slf_loop=True, symmetry_edge=True):
        """Load train data from file and preprocess them.

        Args:
            file_: str, file name for loading data
            slf_loop: bool, if true, add self loop edge for every node
            symmetry_edge: bool, if true, add symmetry edge for every edge

        """
        logging.info('loading data from %s' % file_)
        edge_data_by_type = dict()
        all_edges = list()
        all_nodes = list()

        with open(file_, 'r') as reader:
            for line in reader:
                words = line.strip().split(' ')
                if words[0] not in edge_data_by_type:
                    edge_data_by_type[words[0]] = []
                src, dst = words[1], words[2]
                edge_data_by_type[words[0]].append((src, dst))
                all_edges.append((src, dst))
                all_nodes.append(src)
                all_nodes.append(dst)

                if symmetry_edge:
                    edge_data_by_type[words[0]].append((dst, src))
                    all_edges.append((dst, src))

        all_nodes = list(set(all_nodes))
        all_edges = list(set(all_edges))
        #  edge_data_by_type['Base'] = all_edges

        if slf_loop:
            for e_type in edge_data_by_type.keys():
                for n in all_nodes:
                    edge_data_by_type[e_type].append((n, n))

        # remapping to index
        edges_by_type = {}
        for edge_type, edges in edge_data_by_type.items():
            res_edges = []
            for edge in edges:
                res_edges.append(
                    (self.word2index[edge[0]], self.word2index[edge[1]]))
            edges_by_type[edge_type] = res_edges

        return edges_by_type, all_edges, all_nodes

    def load_test_data(self, file_):
        """Load testing data from file. 
        """
        logging.info('loading data from %s' % file_)

        true_edge_data_by_type = {}
        fake_edge_data_by_type = {}
        with open(file_, 'r') as reader:
            for line in reader:
                words = line.strip().split(' ')
                src, dst = self.word2index[words[1]], self.word2index[words[2]]
                e_type = words[0]
                if int(words[3]) == 1:  # true edges
                    if e_type not in true_edge_data_by_type:
                        true_edge_data_by_type[e_type] = list()
                    true_edge_data_by_type[e_type].append((src, dst))
                else:  # fake edges
                    if e_type not in fake_edge_data_by_type:
                        fake_edge_data_by_type[e_type] = list()
                    fake_edge_data_by_type[e_type].append((src, dst))

        return (true_edge_data_by_type, fake_edge_data_by_type)

    def load_word2index(self):
        """Load words(nodes) from file and map to index.
        """
        word2index = {}
        with open(self.nodes_file, 'r') as reader:
            for index, line in enumerate(reader):
                node = line.strip()
                word2index[node] = index

        return word2index

    def generate_walks(self):
        """Generate random walks for every edge type.
        """
        all_walks = {}
        for e_type in self.edge_types:
            layer_walks = self.simulate_walks(
                edge_type=e_type,
                num_walks=self.config['num_walks'],
                walk_length=self.config['walk_length'])

            all_walks[e_type] = layer_walks

        return all_walks

    def simulate_walks(self, edge_type, num_walks, walk_length, schema=None):
        """Generate random walks in specified edge type.
        """
        walks = []
        nodes = list(range(0, self.graph[edge_type].num_nodes))

        for walk_iter in tqdm.tqdm(range(num_walks)):
            random.shuffle(nodes)
            for node in nodes:
                walk = self.graph[edge_type].random_walk(
                    [node], max_depth=walk_length - 1)
                for i in range(len(walk)):
                    walks.append(walk[i])

        return walks

    def generate_pairs(self, all_walks):
        """Generate word pairs for training.
        """
        logging.info(['edge_types before generate pairs', self.edge_types])

        pairs = []
        skip_window = self.config['win_size'] // 2
        for layer_id, e_type in enumerate(self.edge_types):
            walks = all_walks[e_type]
            for walk in tqdm.tqdm(walks):
                for i in range(len(walk)):
                    for j in range(1, skip_window + 1):
                        if i - j >= 0 and walk[i] != walk[i - j]:
                            neg_nodes = self.graph[e_type].sample_nodes(
                                self.config['neg_num'])
                            pairs.append(
                                (walk[i], walk[i - j], *neg_nodes, layer_id))
                        if i + j < len(walk) and walk[i] != walk[i + j]:
                            neg_nodes = self.graph[e_type].sample_nodes(
                                self.config['neg_num'])
                            pairs.append(
                                (walk[i], walk[i + j], *neg_nodes, layer_id))
        return pairs

    def fetch_batch(self, pairs, batch_size, for_test=False):
        """Produce batch pairs data for training.
        """
        np.random.shuffle(pairs)
        n_batches = (len(pairs) + (batch_size - 1)) // batch_size
        neg_num = len(pairs[0]) - 3

        result = []
        for i in range(1, n_batches):
            batch_pairs = np.array(
                pairs[batch_size * (i - 1):batch_size * i], dtype=np.int64)
            x = batch_pairs[:, 0].reshape(-1, ).astype(np.int64)
            y = batch_pairs[:, 1].reshape(-1, 1, 1).astype(np.int64)
            neg = batch_pairs[:, 2:2 + neg_num].reshape(-1, neg_num,
                                                        1).astype(np.int64)
            t = batch_pairs[:, -1].reshape(-1, 1).astype(np.int64)
            result.append((x, y, neg, t))
        return result


if __name__ == "__main__":
    config = {
        'data_path': './data/youtube/',
        'train_pairs_file': 'train_pairs.pkl',
        'slf_loop': True,
        'symmetry_edge': True,
        'num_walks': 20,
        'walk_length': 10,
        'win_size': 5,
        'neg_num': 5,
    }

    log_format = '%(asctime)s-%(levelname)s-%(name)s: %(message)s'
    logging.basicConfig(level='INFO', format=log_format)

    dataset = Dataset(config)

    logging.info('generating walks')
    all_walks = dataset.generate_walks()
    logging.info('finishing generate walks')
    logging.info(['length of all walks: ', all_walks.keys()])

    train_pairs = dataset.generate_pairs(all_walks)
    pkl.dump(train_pairs,
             open(config['data_path'] + config['train_pairs_file'], 'wb'))
    logging.info('finishing generate train_pairs')
