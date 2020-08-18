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

import numpy as np
import tqdm
import pgl
from pgl.graph_kernel import alias_sample_build_table
from pgl.utils.logger import log

from tokenization import FullTokenizer


def term2id(string, tokenizer, max_seqlen):
    #string = string.split("\t")[1]
    tokens = tokenizer.tokenize(string)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    ids = ids[:max_seqlen-1]
    ids = ids + [2] # ids + [sep]
    ids = ids + [0] * (max_seqlen - len(ids))
    return ids


def dump_graph(args):
    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)
    neg_samples = []
    str2id = dict()
    term_file = io.open(os.path.join(args.outpath, "terms.txt"), "w", encoding=args.encoding)
    terms = []
    count = 0
    item_distribution = []
    edges = []
    train_data = []

    with io.open(args.graphpath, encoding=args.encoding) as f:
        for idx, line in enumerate(f):
            if idx % 100000 == 0:
                log.info("%s readed %s lines" % (args.graphpath, idx))
            slots = []
            for col_idx, col in enumerate(line.strip("\n").split("\t")):
                s = col[:args.max_seqlen]
                if s not in str2id:
                    str2id[s] = count
                    count += 1
                    term_file.write(str(col_idx) + "\t" + col + "\n")
                    item_distribution.append(0)
                slots.append(str2id[s])

            src = slots[0]
            dst = slots[1]
            #neg_samples.append(slots[2:])
            edges.append((src, dst))
            edges.append((dst, src))
            item_distribution[dst] += 1
    edges = np.array(edges, dtype="int64")

    with io.open(args.inpath, encoding=args.encoding) as f:
        for idx, line in enumerate(f):
            if idx % 100000 == 0:
                log.info("%s readed %s lines" % (args.inpath, idx))
            slots = []
            for col_idx, col in enumerate(line.strip("\n").split("\t")):
                s = col[:args.max_seqlen]
                if s not in str2id:
                    str2id[s] = count
                    count += 1
                    term_file.write(str(col_idx) + "\t" + col + "\n")
                    item_distribution.append(0)
                slots.append(str2id[s])

            src = slots[0]
            dst = slots[1]
            neg_samples.append(slots[2:])
            train_data.append((src, dst))
    train_data = np.array(train_data, dtype="int64")

    term_file.close()
    num_nodes = len(str2id)
    str2id.clear()

    log.info("building graph...")
    graph = pgl.graph.Graph(num_nodes=num_nodes, edges=edges)
    indegree = graph.indegree()
    graph.indegree()
    graph.outdegree()
    graph.dump(args.outpath)
    
    # dump alias sample table
    item_distribution = np.array(item_distribution)
    item_distribution = np.sqrt(item_distribution)
    distribution = 1. * item_distribution / item_distribution.sum()
    alias, events = alias_sample_build_table(distribution)
    np.save(os.path.join(args.outpath, "train_data.npy"), train_data)
    np.save(os.path.join(args.outpath, "alias.npy"), alias)
    np.save(os.path.join(args.outpath, "events.npy"), events)
    np.save(os.path.join(args.outpath, "neg_samples.npy"), np.array(neg_samples))
    log.info("End Build Graph")

def dump_node_feat(args):
    log.info("Dump node feat starting...")
    id2str = [line.strip("\n").split("\t")[1] for line in io.open(os.path.join(args.outpath, "terms.txt"), encoding=args.encoding)]
    pool = multiprocessing.Pool()
    tokenizer = FullTokenizer(args.vocab_file)
    term_ids = pool.map(partial(term2id, tokenizer=tokenizer, max_seqlen=args.max_seqlen), id2str)
    np.save(os.path.join(args.outpath, "term_ids.npy"), np.array(term_ids, np.uint16))
    log.info("Dump node feat done.")
    pool.terminate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("-i", "--inpath", type=str, default=None)
    parser.add_argument("-g", "--graphpath", type=str, default=None)
    parser.add_argument("-l", "--max_seqlen", type=int, default=30)
    parser.add_argument("--vocab_file", type=str, default="./vocab.txt")
    parser.add_argument("--encoding", type=str, default="utf8")
    parser.add_argument("-o", "--outpath", type=str, default=None)
    args = parser.parse_args()
    dump_graph(args)
    dump_node_feat(args)
