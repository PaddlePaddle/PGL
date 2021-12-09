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

test_num = 500000000
output_path = "feature_output"
base_dir = "dataset"
prob_dir = output_path

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

h2r_prob = pickle.load(open(prob_dir + "/h2r_prob.pkl", "rb"))
r2t_prob = pickle.load(open(prob_dir + "/r2t_prob.pkl", "rb"))
r2h_prob = pickle.load(open(prob_dir + "/r2h_prob.pkl", "rb"))

r2t_h2r = np.zeros((1315, 1315), dtype=np.float16)
for i in tqdm(range(1315)):
    for t in r2t_prob[i]:
        prob = r2t_prob[i][t]
        if t not in h2r_prob:
            continue
        for r in h2r_prob[t]:
            prob2 = h2r_prob[t][r]
            r2t_h2r[i, r] += prob * prob2
#np.save("%s/test_feats/r2t_h2r.npy" % output_path, r2t_h2r)


def get_r2t_h2r_feat(t_candidate, hr, path):
    r2t_h2r_feat = np.zeros(t_candidate.shape, dtype=np.float16)
    for i in tqdm(range(t_candidate.shape[0])):
        r1 = hr[i, 1]
        for j in range(t_candidate.shape[1]):
            tail = t_candidate[i, j]
            if tail in h2r_prob:
                for r2 in h2r_prob[tail]:
                    prob = r2t_h2r[r1, r2] * r2h_prob[r2][tail]
                    r2t_h2r_feat[i, j] += prob
    np.save(path, r2t_h2r_feat)
    return r2t_h2r_feat


print("load done")

get_r2t_h2r_feat(val_t_candidate, val_hr,
                 "%s/valid_feats/r2t_h2r_feat.npy" % output_path)
print("valid_done")
get_r2t_h2r_feat(test_t_candidate, test_hr,
                 "%s/test_feats/r2t_h2r_feat.npy" % output_path)
