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
Data pre-processing for metapath2vec model.
"""

import os
import sys
import tqdm
import time
import logging
import random
import argparse
import numpy as np
import pickle as pkl

from pgl.utils.logger import log
from utils.config import prepare_config, make_dir

# name  ID  g_index


def remapping_id(file_, start_index, node_type, separator="\t"):
    """Mapp the ID and name of nodes to index.
    """
    node_types = []
    id2index = {}
    name2index = {}
    index = start_index
    with open(file_, encoding="ISO-8859-1") as reader:
        for line in reader:
            tokens = line.strip().split(separator)
            id2index[tokens[0]] = str(index)
            if len(tokens) == 2:
                name2index[tokens[1]] = str(index)
            node_types.append((str(index), node_type))
            index += 1

    return id2index, name2index, node_types


def load_edges(file_, src2index, dst2index, symmetry=False):
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


def load_label(file_, name2index):
    index_label = []
    with open(file_, encoding="ISO-8859-1") as reader:
        for line in reader:
            tokens = line.strip().split(' ')
            name, label = tokens[0], int(tokens[1]) - 1
            if name in name2index:
                index_label.append((name2index[name], str(label)))

    return index_label


def main(config):
    conf_id2index, conf_name2index, conf_node_type = remapping_id(
        os.path.join(config.data_path, 'id_conf.txt'),
        start_index=0,
        node_type='c')
    log.info('%d venues have been loaded.' % (len(conf_id2index)))

    author_id2index, author_name2index, author_node_type = remapping_id(
        os.path.join(config.data_path, 'id_author.txt'),
        start_index=len(conf_id2index),
        node_type='a')
    log.info('%d authors have been loaded.' % (len(author_id2index)))

    paper_id2index, paper_name2index, paper_node_type = remapping_id(
        os.path.join(config.data_path, 'paper.txt'),
        start_index=(len(conf_id2index) + len(author_id2index)),
        node_type='p',
        separator='\t')
    log.info('%d papers have been loaded.' % (len(paper_id2index)))

    node_types = conf_node_type + author_node_type + paper_node_type

    paper2author_edges = load_edges(
        os.path.join(config.data_path, 'paper_author.txt'), paper_id2index,
        author_id2index)
    log.info('%d paper2author edges have been loaded.' %
             (len(paper2author_edges)))

    paper2conf_edges = load_edges(
        os.path.join(config.data_path, 'paper_conf.txt'), paper_id2index,
        conf_id2index)
    log.info('%d paper2conf edges have been loaded.' % (len(paper2conf_edges)))

    author_label = load_label(config.author_label_file, author_name2index)
    conf_label = load_label(config.venue_label_file, conf_name2index)

    make_dir(config.processed_path)
    node_types_file = os.path.join(config.processed_path, 'node_types.txt')
    log.info("saving node_types to %s" % node_types_file)
    with open(node_types_file, 'w') as writer:
        for item in tqdm.tqdm(node_types):
            writer.write("%s\t%s\n" % (item[1], item[0]))

    p2a_edges_file = os.path.join(config.processed_path,
                                  'paper2author_edges.txt')
    log.info("saving paper2author edges to %s" % p2a_edges_file)
    with open(p2a_edges_file, 'w') as writer:
        for item in tqdm.tqdm(paper2author_edges):
            writer.write("\t".join(item) + "\n")

    p2c_edges_file = os.path.join(config.processed_path,
                                  'paper2conf_edges.txt')
    log.info("saving paper2conf edges to %s" % p2c_edges_file)
    with open(p2c_edges_file, 'w') as writer:
        for item in tqdm.tqdm(paper2conf_edges):
            writer.write("\t".join(item) + "\n")

    author_label_file = os.path.join(config.processed_path, 'author_label.txt')
    log.info("saving author label to %s" % author_label_file)
    with open(author_label_file, 'w') as writer:
        for item in tqdm.tqdm(author_label):
            writer.write("\t".join(item) + "\n")

    conf_label_file = os.path.join(config.processed_path, 'conf_label.txt')
    log.info("saving conf label to %s" % conf_label_file)
    with open(conf_label_file, 'w') as writer:
        for item in tqdm.tqdm(conf_label):
            writer.write("\t".join(item) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='metapath2vec')
    parser.add_argument('--config', default="./config.yaml", type=str)
    args = parser.parse_args()

    config = prepare_config(args.config)

    main(config)
