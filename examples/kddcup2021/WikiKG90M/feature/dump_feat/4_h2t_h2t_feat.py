#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
#
# File: 4_h2t_h2t_feat.py
# Author: suweiyue(suweiyue@baidu.com)
# Date: 2021/06/03 23:12:20
#
########################################################################
"""
    Comment.
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import sys
import argparse
import logging
import numpy as np
import pickle
from scipy.sparse import lil_matrix
from tqdm import tqdm
import math
from multiprocessing import Pool

test_num = 50000000
base_dir = "dataset"
output_path = "feature_output"
val_t_correct_index = np.load(
    base_dir + "/wikikg90m_kddcup2021/processed/val_t_correct_index.npy",
    mmap_mode="r")
train_hrt = np.load(
    base_dir + "/wikikg90m_kddcup2021/processed/train_hrt.npy", mmap_mode="r")
val_hr = np.load(
    base_dir + "/wikikg90m_kddcup2021/processed/val_hr.npy", mmap_mode="r")
val_t_candidate = np.load(
    base_dir + "/wikikg90m_kddcup2021/processed/val_t_candidate.npy",
    mmap_mode="r")
test_hr = np.load(
    base_dir + "/wikikg90m_kddcup2021/processed/test_hr.npy",
    mmap_mode="r")[:test_num]
test_t_candidate = np.load(
    base_dir + "/wikikg90m_kddcup2021/processed/test_t_candidate.npy",
    mmap_mode="r")[:test_num]

prob_dir = output_path

h2t_prob = pickle.load(open(prob_dir + "/h2t_prob.pkl", "rb"))
t2h_prob = pickle.load(open(prob_dir + "/t2h_prob.pkl", "rb"))


def get_h2t_h2t_feat(t_candidate, hr, path):
    h2t_h2t_feat = np.zeros(t_candidate.shape, dtype=np.float16)
    for i in tqdm(range(t_candidate.shape[0])):
        h = hr[i, 0]
        if h not in h2t_prob:
            continue
        for j in range(t_candidate.shape[1]):
            tail = t_candidate[i, j]
            if tail not in t2h_prob:
                continue
            for e in h2t_prob[h]:
                if e not in t2h_prob[tail]:
                    continue
                prob = h2t_prob[h][e] * h2t_prob[e][tail]
                h2t_h2t_feat[i][j] += prob
    np.save(path, h2t_h2t_feat)
    return h2t_h2t_feat


print("load done")

get_h2t_h2t_feat(val_t_candidate, val_hr,
                 "%s/valid_feats/h2t_h2t_feat.npy" % output_path)
print("valid_done")
get_h2t_h2t_feat(test_t_candidate, test_hr,
                 "%s/test_feats/h2t_h2t_feat.npy" % output_path)
