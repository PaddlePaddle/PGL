#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
# File: dump_graph.py
# Author: suweiyue(suweiyue@baidu.com)
# Date: 2020/03/01 22:17:13
#
########################################################################
"""
    Comment.
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
#from __future__ import unicode_literals

import io
import os
import sys
import argparse
import logging
import multiprocessing
from functools import partial
from io import open
import logging

import numpy as np
import tqdm
import pgl
from pgl.graph_kernel import alias_sample_build_table
from pgl.utils.logger import log
import paddle.fluid.dygraph as D
import paddle.fluid as F
from easydict import EasyDict as edict
import yaml

from ernie.tokenizing_ernie import ErnieTokenizer
from ernie.tokenizing_ernie import ErnieTinyTokenizer
from ernie.modeling_ernie import ErnieModel

log.setLevel(logging.DEBUG)


def term2id(string, tokenizer, max_seqlen):
    tokens = tokenizer.tokenize(string)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    ids = ids[:max_seqlen - 1]
    ids = ids + [tokenizer.sep_id]  # ids + [sep]
    ids = ids + [tokenizer.pad_id] * (max_seqlen - len(ids))
    return ids


def load_graph(config, str2id, term_file, terms, item_distribution):
    edges = []
    with io.open(config.graph_data, encoding=config.encoding) as f:
        for idx, line in enumerate(f):
            if idx % 100000 == 0:
                log.info("%s readed %s lines" % (config.graph_data, idx))
            slots = []
            for col_idx, col in enumerate(line.strip("\n").split("\t")):
                s = col[:config.max_seqlen]
                if s not in str2id:
                    str2id[s] = len(str2id)
                    term_file.write(str(col_idx) + "\t" + col + "\n")
                    item_distribution.append(0)
                slots.append(str2id[s])

            src = slots[0]
            dst = slots[1]
            edges.append((src, dst))
            edges.append((dst, src))
            item_distribution[dst] += 1
    edges = np.array(edges, dtype="int64")
    return edges


def load_link_predict_train_data(config, str2id, term_file, terms,
                                 item_distribution):
    train_data = []
    neg_samples = []
    with io.open(config.train_data, encoding=config.encoding) as f:
        for idx, line in enumerate(f):
            if idx % 100000 == 0:
                log.info("%s readed %s lines" % (config.train_data, idx))
            slots = []
            for col_idx, col in enumerate(line.strip("\n").split("\t")):
                s = col[:config.max_seqlen]
                if s not in str2id:
                    str2id[s] = len(str2id)
                    term_file.write(str(col_idx) + "\t" + col + "\n")
                    item_distribution.append(0)
                slots.append(str2id[s])

            src = slots[0]
            dst = slots[1]
            neg_samples.append(slots[2:])
            train_data.append((src, dst))
    train_data = np.array(train_data, dtype="int64")
    np.save(os.path.join(config.graph_work_path, "train_data.npy"), train_data)
    if len(neg_samples) != 0:
        np.save(
            os.path.join(config.graph_work_path, "neg_samples.npy"),
            np.array(neg_samples))


def load_node_classification_train_data(config, str2id, term_file, terms,
                                        item_distribution):
    train_data = []
    neg_samples = []
    with io.open(config.train_data, encoding=config.encoding) as f:
        for idx, line in enumerate(f):
            if idx % 100000 == 0:
                log.info("%s readed %s lines" % (config.train_data, idx))
            slots = []
            col_idx = 0
            slots = line.strip("\n").split("\t")
            col = slots[0]
            label = int(slots[1])
            text = col[:config.max_seqlen]
            if text not in str2id:
                str2id[text] = len(str2id)
                term_file.write(str(col_idx) + "\t" + col + "\n")
                item_distribution.append(0)
            src = str2id[text]
            train_data.append([src, label])
    train_data = np.array(train_data, dtype="int64")
    np.save(os.path.join(config.graph_work_path, "train_data.npy"), train_data)


def dump_graph(config):
    if not os.path.exists(config.graph_work_path):
        os.makedirs(config.graph_work_path)
    str2id = dict()
    term_file = io.open(
        os.path.join(config.graph_work_path, "terms.txt"),
        "w",
        encoding=config.encoding)
    terms = []
    item_distribution = []

    edges = load_graph(config, str2id, term_file, terms, item_distribution)
    #load_train_data(config, str2id, term_file, terms, item_distribution)
    if config.task == "link_predict":
        load_link_predict_train_data(config, str2id, term_file, terms,
                                     item_distribution)
    elif config.task == "node_classification":
        load_node_classification_train_data(config, str2id, term_file, terms,
                                            item_distribution)
    else:
        raise ValueError

    term_file.close()
    num_nodes = len(str2id)
    str2id.clear()

    log.info("building graph...")
    graph = pgl.graph.Graph(num_nodes=num_nodes, edges=edges)
    indegree = graph.indegree()
    graph.indegree()
    graph.outdegree()
    graph.dump(config.graph_work_path)

    # dump alias sample table
    item_distribution = np.array(item_distribution)
    item_distribution = np.sqrt(item_distribution)
    distribution = 1. * item_distribution / item_distribution.sum()
    alias, events = alias_sample_build_table(distribution)
    np.save(os.path.join(config.graph_work_path, "alias.npy"), alias)
    np.save(os.path.join(config.graph_work_path, "events.npy"), events)
    log.info("End Build Graph")


def dump_node_feat(config):
    log.info("Dump node feat starting...")
    id2str = [
        line.strip("\n").split("\t")[-1]
        for line in io.open(
            os.path.join(config.graph_work_path, "terms.txt"),
            encoding=config.encoding)
    ]
    if "tiny" in config.ernie_name:
        tokenizer = ErnieTinyTokenizer.from_pretrained(config.ernie_name)
        #tokenizer.vocab = tokenizer.sp_model.vocab
        term_ids = [
            partial(
                term2id, tokenizer=tokenizer, max_seqlen=config.max_seqlen)(s)
            for s in id2str
        ]
    else:
        tokenizer = ErnieTokenizer.from_pretrained(config.ernie_name)
        pool = multiprocessing.Pool()
        term_ids = pool.map(partial(
            term2id, tokenizer=tokenizer, max_seqlen=config.max_seqlen),
                            id2str)
        pool.terminate()
    node_feat_path = os.path.join(config.graph_work_path, "node_feat")
    if not os.path.exists(node_feat_path):
        os.makedirs(node_feat_path)
    np.save(
        os.path.join(config.graph_work_path, "node_feat", "term_ids.npy"),
        np.array(term_ids, np.uint16))
    log.info("Dump node feat done.")


def download_ernie_model(config):
    place = F.CUDAPlace(0)
    with D.guard(place):
        model = ErnieModel.from_pretrained(config.ernie_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--conf", type=str, default="./config.yaml")
    args = parser.parse_args()
    config = edict(yaml.load(open(args.conf), Loader=yaml.FullLoader))

    dump_graph(config)
    dump_node_feat(config)
    download_ernie_model(config)
