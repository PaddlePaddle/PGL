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
This file implement the sampler to sample metapath random walk sequence for 
training metapath2vec model.
"""

import multiprocessing
from multiprocessing import Pool
import argparse
import sys
import os
import numpy as np
import pickle as pkl
import tqdm
import time
import logging
import random
from pgl.contrib import heter_graph
from pgl.sample import metapath_randomwalk
from utils import *


class Sampler(object):
    """Implemetation of sampler in order to sample metapath random walk.

    Args:
        config: dict, some configure parameters.
    """

    def __init__(self, config):
        self.config = config
        self.build_graph()

    def build_graph(self):
        """Build pgl heterogeneous graph.
        """
        self.conf_id2index, self.conf_name2index, conf_node_type = self.remapping_id(
            self.config['data_path'] + 'id_conf.txt',
            start_index=0,
            node_type='conf')
        logging.info('%d venues have been loaded.' % (len(self.conf_id2index)))

        self.author_id2index, self.author_name2index, author_node_type = self.remapping_id(
            self.config['data_path'] + 'id_author.txt',
            start_index=len(self.conf_id2index),
            node_type='author')
        logging.info('%d authors have been loaded.' %
                     (len(self.author_id2index)))

        self.paper_id2index, self.paper_name2index, paper_node_type = self.remapping_id(
            self.config['data_path'] + 'paper.txt',
            start_index=(len(self.conf_id2index) + len(self.author_id2index)),
            node_type='paper',
            separator='\t')
        logging.info('%d papers have been loaded.' %
                     (len(self.paper_id2index)))

        node_types = conf_node_type + author_node_type + paper_node_type
        num_nodes = len(node_types)
        edges_by_types = {}
        paper_author_edges = self.load_edges(
            self.config['data_path'] + 'paper_author.txt', self.paper_id2index,
            self.author_id2index)
        paper_conf_edges = self.load_edges(
            self.config['data_path'] + 'paper_conf.txt', self.paper_id2index,
            self.conf_id2index)

        edges_by_types['edge'] = paper_author_edges + paper_conf_edges
        logging.info('%d edges have been loaded.' %
                     (len(edges_by_types['edge'])))

        node_features = {
            'index': np.array([i for i in range(num_nodes)]).reshape(
                -1, 1).astype(np.int64)
        }

        self.graph = heter_graph.HeterGraph(
            num_nodes=num_nodes,
            edges=edges_by_types,
            node_types=node_types,
            node_feat=node_features)

    def remapping_id(self, file_, start_index, node_type, separator='\t'):
        """Mapp the ID and name of nodes to index.
        """
        node_types = []
        id2index = {}
        name2index = {}
        index = start_index
        with open(file_, encoding="ISO-8859-1") as reader:
            for line in reader:
                tokens = line.strip().split(separator)
                id2index[tokens[0]] = index
                if len(tokens) == 2:
                    name2index[tokens[1]] = index
                node_types.append((index, node_type))
                index += 1

        return id2index, name2index, node_types

    def load_edges(self, file_, src2index, dst2index, symmetry=True):
        """Load edges from file.
        """
        edges = []
        with open(file_, 'r') as reader:
            for line in reader:
                items = line.strip().split()
                src, dst = src2index[items[0]], dst2index[items[1]]
                edges.append((src, dst))
                if symmetry:
                    edges.append((dst, src))
            edges = list(set(edges))
        return edges

    def generate_multi_class_data(self, name_label_file):
        """Mapp the data that will be used in multi class task to index.
        """
        if 'author' in name_label_file:
            name2index = self.author_name2index
        else:
            name2index = self.conf_name2index

        index_label_list = []
        with open(name_label_file, encoding="ISO-8859-1") as reader:
            for line in reader:
                tokens = line.strip().split(' ')
                name, label = tokens[0], int(tokens[1])
                index = name2index[name]
                index_label_list.append((index, label))

        return index_label_list


def generate_walks(args):
    """Generate metapath random walk and save to file.
    """
    g, meta_path, filename, walk_length = args
    walks = []
    node_types = g._node_types
    first_type = meta_path.split('-')[0]
    nodes = np.where(node_types == first_type)[0]
    if len(nodes) > 4000:
        nodes = np.random.choice(nodes, 4000, replace=False)

    logging.info('%d number of start nodes' % (len(nodes)))
    logging.info('save walks in file: %s' % (filename))

    with open(filename, 'w') as writer:
        for start_node in nodes:
            walk = metapath_randomwalk(g, start_node, meta_path, walk_length)
            walk = [str(walk[i]) for i in range(0, len(walk), 2)]  # skip paper
            writer.write(' '.join(walk) + '\n')


def multiprocess_generate_walks(sampler, edge_type, meta_path, num_walks,
                                walk_length, saved_path):
    """Use multiprocess to generate metapath random walk.
    """
    args = []
    for i in range(num_walks):
        filename = saved_path + '%04d' % (i)
        args.append(
            (sampler.graph[edge_type], meta_path, filename, walk_length))

    pool = Pool(16)
    pool.map(generate_walks, args)
    pool.close()
    pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='metapath2vec')
    parser.add_argument(
        '-c',
        '--config',
        default=None,
        type=str,
        help='config file path (default: None)')
    args = parser.parse_args()

    if args.config:
        # load config file
        config = Config(args.config, isCreate=False, isSave=False)
        config = config()
        config = config['sampler']['args']
    else:
        raise AssertionError(
            "Configuration file need to be specified. Add '-c config.yaml', for example."
        )

    log_format = '%(asctime)s-%(levelname)s-%(name)s: %(message)s'
    logging.basicConfig(level="INFO", format=log_format)

    logging.info(config)

    log_format = '%(asctime)s-%(levelname)s-%(name)s: %(message)s'
    logging.basicConfig(level=getattr(logging, 'INFO'), format=log_format)

    if not os.path.exists(config['output_path']):
        os.makedirs(config['output_path'])

    config['walk_saved_path'] = config['output_path'] + config[
        'walk_saved_path']
    if not os.path.exists(config['walk_saved_path']):
        os.makedirs(config['walk_saved_path'])

    sampler = Sampler(config)

    begin = time.time()
    logging.info('multi process sampling')
    multiprocess_generate_walks(
        sampler=sampler,
        edge_type='edge',
        meta_path=config['metapath'],
        num_walks=config['num_walks'],
        walk_length=config['walk_length'],
        saved_path=config['walk_saved_path'])
    logging.info('total time: %.4f' % (time.time() - begin))

    logging.info('generating multi class data')
    word_label_list = sampler.generate_multi_class_data(config[
        'author_label_file'])
    with open(config['output_path'] + config['new_author_label_file'],
              'w') as writer:
        for line in word_label_list:
            line = [str(i) for i in line]
            writer.write(' '.join(line) + '\n')

    word_label_list = sampler.generate_multi_class_data(config[
        'venue_label_file'])
    with open(config['output_path'] + config['new_venue_label_file'],
              'w') as writer:
        for line in word_label_list:
            line = [str(i) for i in line]
            writer.write(' '.join(line) + '\n')
    logging.info('finished')
