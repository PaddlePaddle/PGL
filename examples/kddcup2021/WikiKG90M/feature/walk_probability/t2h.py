#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
#
# File: a.py
# Author: suweiyue(suweiyue@baidu.com)
# Date: 2021/04/13 21:58:37
#
########################################################################
"""
    Comment.
"""

import numpy as np
import pickle
from tqdm import tqdm


train_hrt = np.load("./dataset/wikikg90m_kddcup2021/processed/train_hrt.npy", mmap_mode="r")
data = dict()

for h, r, t in tqdm(train_hrt):
    if t not in data:
        data[t] = dict()
    if not h in data[t]:
        data[t][h] = 1.
    else:
        data[t][h] += 1.

#pickle.dump(data, open("train_valid/t2h.pkl", "wb"))

del train_hrt

for t in tqdm(data):
    t_sum = sum(data[t].values())
    for h in data[t]:
        data[t][h] /= t_sum

pickle.dump(data, open("feature_output/t2h_prob.pkl", "wb"))
