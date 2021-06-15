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


# HT
def f(x):
    res = np.zeros_like(x)
    unique, counts = np.unique(x, return_counts=True)
    mapper_dict = {}
    for idx, count in zip(unique, counts):
        mapper_dict[idx] = count

    def mp(entry):
        return mapper_dict[entry]

    mp = np.vectorize(mp)
    return mp(x)


# valid
val_h_sorted_index = np.argsort(val_hr[:, 0], axis=0)
val_h_sorted = val_hr[val_h_sorted_index]
val_h_sorted_index_part = []
last_start = -1
tmp = []
for i in tqdm(range(len(val_h_sorted) + 1)):
    if i == len(val_h_sorted):
        val_h_sorted_index_part.append(tmp)
        break
    if val_h_sorted[i][0] > last_start:
        if last_start != -1:
            val_h_sorted_index_part.append(tmp)
        tmp = []
        last_start = val_h_sorted[i][0]
    tmp.append(i)
val_h_sorted_index_arr = [
    np.array(
        idx, dtype="int32") for idx in val_h_sorted_index_part
]
inputs = [
    val_t_candidate[val_h_sorted_index[arr]] for arr in val_h_sorted_index_arr
]
mapped_array = None
with Pool(20) as p:
    mapped_array = list(tqdm(p.imap(f, inputs), total=len(inputs)))
ht_feat = np.zeros_like(val_t_candidate)
for (arr, mapped) in zip(val_h_sorted_index_arr, mapped_array):
    ht_feat[val_h_sorted_index[arr]] = mapped
np.save("%s/valid_feats/ht_feat.npy" % output_path, ht_feat.astype(np.float32))

# test
test_h_sorted_index = np.argsort(test_hr[:, 0], axis=0)
test_h_sorted = test_hr[test_h_sorted_index]
test_h_sorted_index_part = []
last_start = -1
tmp = []
for i in tqdm(range(len(test_h_sorted) + 1)):
    if i == len(test_h_sorted):
        test_h_sorted_index_part.append(tmp)
        break
    if test_h_sorted[i][0] > last_start:
        if last_start != -1:
            test_h_sorted_index_part.append(tmp)
        tmp = []
        last_start = test_h_sorted[i][0]
    tmp.append(i)
test_h_sorted_index_arr = [
    np.array(
        idx, dtype="int32") for idx in test_h_sorted_index_part
]
inputs = [
    test_t_candidate[test_h_sorted_index[arr]]
    for arr in test_h_sorted_index_arr
]
mapped_array = None
with Pool(20) as p:
    mapped_array = list(tqdm(p.imap(f, inputs), total=len(inputs)))
ht_feat = np.zeros_like(test_t_candidate)
for (arr, mapped) in zip(test_h_sorted_index_arr, mapped_array):
    ht_feat[test_h_sorted_index[arr]] = mapped
np.save("%s/test_feats/ht_feat.npy" % output_path, ht_feat.astype(np.float32))
