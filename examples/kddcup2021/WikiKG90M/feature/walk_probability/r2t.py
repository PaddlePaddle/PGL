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
data = [dict() for _ in range(1315)]

for h, r, t in tqdm(train_hrt):
    if not t in data[r]:
        data[r][t] = 1
    else:
        data[r][t] += 1

for r in range(1315):
    r_sum = sum(data[r].values())
    for t in data[r]:
        data[r][t] /= r_sum


pickle.dump(data, open("feature_output/r2t_prob.pkl", "wb"))
