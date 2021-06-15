# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
#
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
test_num = 500000000

base_dir = "dataset"
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

r2t_prob = pickle.load(open(prob_dir + "/r2t_prob.pkl", "rb"))
t2r_prob = pickle.load(open(prob_dir + "/t2r_prob.pkl", "rb"))

print("load data done")
rrt = np.zeros((1315, 1315))
for i in tqdm(range(1315)):
    for t in r2t_prob[i]:
        prob = r2t_prob[i][t]
        for r in t2r_prob[t]:
            prob2 = t2r_prob[t][r]
            rrt[i, r] += prob * prob2
#np.save("%s/test_feats/rrt_new.npy" % output_path, rrt)


def get_rrt_feat(t_candidate, hr, path):
    rrt_feat = np.zeros(t_candidate.shape, dtype=np.float16)
    for i in tqdm(range(t_candidate.shape[0])):
        r1 = hr[i, 1]
        for j in range(t_candidate.shape[1]):
            tail = t_candidate[i, j]
            if tail in t2r_prob:
                for r2 in t2r_prob[tail]:
                    prob = rrt[r1, r2] * r2t_prob[r2][tail]
                    rrt_feat[i, j] += prob
    np.save(path, rrt_feat)


get_rrt_feat(val_t_candidate, val_hr,
             "%s/valid_feats/rrt_feat.npy" % output_path)
print("valid done")
get_rrt_feat(test_t_candidate, test_hr,
             "%s/test_feats/rrt_feat.npy" % output_path)
print("test done")
