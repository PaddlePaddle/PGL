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

test_num = 50000000
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

h2t_prob = pickle.load(open(prob_dir + "/h2t_prob.pkl", "rb"))
t2h_prob = pickle.load(open(prob_dir + "/t2h_prob.pkl", "rb"))

print("load data done")

hh_byt_prob = dict()
for h in tqdm(h2t_prob):
    if len(h2t_prob[h]) > 10:
        continue
    for t in h2t_prob[h]:
        prob = h2t_prob[h][t]
        if len(t2h_prob[t]) > 10:
            continue
        for h2 in t2h_prob[t]:
            prob2 = t2h_prob[t][h2]
            if h not in hh_byt_prob:
                hh_byt_prob[h] = dict()
            if h2 not in hh_byt_prob[h]:
                hh_byt_prob[h][h2] = prob * prob2
            else:
                hh_byt_prob[h][h2] += prob * prob2


def get_hht_feat(t_candidate, hr, path):
    hht_feat = np.zeros(t_candidate.shape, dtype=np.float16)
    for i in tqdm(range(t_candidate.shape[0])):
        h1 = hr[i, 0]
        if h1 not in hh_byt_prob:
            continue
        for j in range(t_candidate.shape[1]):
            tail = t_candidate[i, j]
            if tail in t2h_prob:
                for h2 in t2h_prob[tail]:
                    if h2 not in hh_byt_prob[h1]:
                        continue
                    prob = hh_byt_prob[h1][h2] * h2t_prob[h2][tail]
                    hht_feat[i, j] += prob
    np.save(path, hht_feat)
    return hht_feat


get_hht_feat(val_t_candidate, val_hr,
             "%s/valid_feats/hht_feat.npy" % output_path)
print("valid done")
get_hht_feat(test_t_candidate, test_hr,
             "%s/test_feats/hht_feat.npy" % output_path)
print("test done")
