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

r2t_prob = pickle.load(open(prob_dir + "/r2t_prob.pkl", "rb"))

print("load data done")


# 7. r2t_feat r2t 
def get_r2t_feat(t_candidate, hr, path):
    r2t_feat = np.zeros(t_candidate.shape, dtype=np.float16)
    for i in tqdm(range(t_candidate.shape[0])):
        r = hr[i, 1]
        for j in range(t_candidate.shape[1]):
            t = t_candidate[i, j]
            if t in r2t_prob[r]:
                r2t_feat[i, j] = r2t_prob[r][t]
    np.save(path, r2t_feat)
    return r2t_feat


get_r2t_feat(val_t_candidate, val_hr,
             "%s/valid_feats/r2t_feat.npy" % output_path)
print("valid done")
get_r2t_feat(test_t_candidate, test_hr,
             "%s/test_feats/r2t_feat.npy" % output_path)
print("test done")
