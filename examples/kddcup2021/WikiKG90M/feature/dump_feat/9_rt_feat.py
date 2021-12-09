#!/usr/bin/env python
# coding: utf-8

from multiprocessing import Pool
from tqdm import tqdm

import numpy as np

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


def f(x):
    unique, counts = np.unique(x, return_counts=True)
    mapper_dict = {}
    for idx, count in zip(unique, counts):
        mapper_dict[idx] = count

    def mp(entry):
        return mapper_dict[entry]

    mp = np.vectorize(mp)
    return mp(x)


val_r_sorted_index = np.argsort(val_hr[:, 1], axis=0)
val_r_sorted = val_hr[val_r_sorted_index]
val_r_sorted_index_part = []
last_start = -1
tmp = []
for i in tqdm(range(len(val_r_sorted) + 1)):
    if i == len(val_r_sorted):
        val_r_sorted_index_part.append(tmp)
        break
    if val_r_sorted[i][1] > last_start:
        if last_start != -1:
            val_r_sorted_index_part.append(tmp)
        tmp = []
        last_start = val_r_sorted[i][1]
    tmp.append(i)
val_r_sorted_index_arr = [
    np.array(
        idx, dtype="int32") for idx in val_r_sorted_index_part
]
inputs = [
    val_t_candidate[val_r_sorted_index[arr]] for arr in val_r_sorted_index_arr
]
mapped_array = None
with Pool(20) as p:
    mapped_array = list(tqdm(p.imap(f, inputs), total=len(inputs)))
rt_feat = np.zeros_like(val_t_candidate, dtype=np.float32)
for (arr, mapped) in zip(val_r_sorted_index_arr, mapped_array):
    rt_feat[val_r_sorted_index[arr]] = mapped
np.save("%s/valid_feats/rt_feat.npy" % output_path, rt_feat.astype(np.float32))

test_r_sorted_index = np.argsort(test_hr[:, 1], axis=0)
test_r_sorted = test_hr[test_r_sorted_index]
test_r_sorted_index_part = []
last_start = -1
tmp = []
for i in tqdm(range(len(test_r_sorted) + 1)):
    if i == len(test_r_sorted):
        test_r_sorted_index_part.append(tmp)
        break
    if test_r_sorted[i][1] > last_start:
        if last_start != -1:
            test_r_sorted_index_part.append(tmp)
        tmp = []
        last_start = test_r_sorted[i][1]
    tmp.append(i)
test_r_sorted_index_arr = [
    np.array(
        idx, dtype="int32") for idx in test_r_sorted_index_part
]
inputs = [
    test_t_candidate[test_r_sorted_index[arr]]
    for arr in test_r_sorted_index_arr
]
mapped_array = None
with Pool(20) as p:
    mapped_array = list(tqdm(p.imap(f, inputs), total=len(inputs)))
rt_feat = np.zeros_like(test_t_candidate, dtype=np.float32)
for (arr, mapped) in zip(test_r_sorted_index_arr, mapped_array):
    rt_feat[test_r_sorted_index[arr]] = mapped
np.save("%s/test_feats/rt_feat.npy" % output_path, rt_feat.astype(np.float32))
