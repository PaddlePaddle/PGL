#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
#
# File: valid_score.py
# Author: suweiyue(suweiyue@baidu.com)
# Date: 2021/04/19 14:09:25
#
########################################################################
"""
    Comment.
"""
import numpy as np
from glob import glob
from tqdm import tqdm

import sys
import argparse
import logging
from multiprocessing import Pool


def process_func(path):
    data = []
    for idx, line in enumerate(open(path)):
        fields = line.strip("\n").split(" ")
        idx = int(fields[0].split("\t")[0])
        fields[0] = fields[0].split("\t")[-1].split(";")[-1]
        value = np.array(fields, dtype=np.float16)
        data.append((idx, value))
    return data


def load_emb(entity_emb, path):
    entity_num = 87143637
    paths = glob(args.data_path + "/*")
    with Pool(20) as p:
        rets = p.map(process_func, paths)
    data = np.zeros((entity_num, 768), dtype=np.float16)
    for ret in rets:
        for idx, value in ret:
            data[idx] = value
    np.save(path, data)
    return data


def main(args):
    load_emb(args.data_path, args.tmp_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--data_path", type=str, default="./abcd")
    parser.add_argument("--tmp_path", type=str, default="entity_emb_19.npy")
    parser.add_argument("--output_path", type=str, default="entity_emb_19.npy")
    parser.add_argument("--relation_path", type=str, default=None)
    parser.add_argument("--score_func", type=str, default="TransE")
    args = parser.parse_args()
    main(args)
