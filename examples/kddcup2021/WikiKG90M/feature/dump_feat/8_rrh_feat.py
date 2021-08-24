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
from tqdm import tqdm
import math
from multiprocessing import Pool

output_path = "feature_output"
base_dir = "dataset"
prob_dir = output_path
test_num = 50000

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

print("load data done")
# 8. rrh_feat r2h h2r

h2r_prob = pickle.load(open(prob_dir + "/h2r_prob.pkl", "rb"))
r2h_prob = pickle.load(open(prob_dir + "/r2h_prob.pkl", "rb"))
rrh = np.zeros((1315, 1315))
for i in tqdm(range(1315)):
    for h in r2h_prob[i]:
        prob = r2h_prob[i][h]
        for r in h2r_prob[h]:
            prob2 = h2r_prob[h][r]
            rrh[i, r] += prob * prob2

r2t_prob = pickle.load(open(prob_dir + "/r2t_prob.pkl", "rb"))
t2r_prob = pickle.load(open(prob_dir + "/t2r_prob.pkl", "rb"))


def get_rrh_feat(t_candidate, hr, path):
    rrh_feat = np.zeros(t_candidate.shape, dtype=np.float16)
    for i in tqdm(range(t_candidate.shape[0])):
        r1 = hr[i, 1]
        for j in range(t_candidate.shape[1]):
            tail = t_candidate[i, j]
            if tail in t2r_prob:
                for r2 in t2r_prob[tail]:
                    if tail not in r2t_prob[r2]:
                        print(r1, r2, tail)
                        exit()
                    prob = rrh[r1, r2] * r2t_prob[r2][tail]
                    rrh_feat[i, j] += prob
    np.save(path, rrh_feat)


get_rrh_feat(val_t_candidate, val_hr,
             "%s/valid_feats/rrh_feat.npy" % output_path)
print("valid done")
get_rrh_feat(test_t_candidate, test_hr,
             "%s/test_feats/rrh_feat.npy" % output_path)
print("test done")
