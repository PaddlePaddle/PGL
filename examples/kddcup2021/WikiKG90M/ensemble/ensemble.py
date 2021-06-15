#!/usr/bin/env python
# coding: utf-8

import pgl
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
from ogb.lsc import WikiKG90MDataset, WikiKG90MEvaluator

postfix = ""
main_dir = "dataset%s/wikikg90m_kddcup2021/processed/" % postfix

val_t_correct_index = np.load(
    main_dir + "val_t_correct_index.npy", mmap_mode="r")
train_hrt = np.load(main_dir + "train_hrt.npy", mmap_mode="r")
val_hr = np.load(main_dir + "val_hr.npy", mmap_mode="r")
val_hrt = np.load(main_dir + "val_hrt.npy", mmap_mode="r")
val_t_candidate = np.load(main_dir + "val_t_candidate.npy", mmap_mode="r")
test_t_candidate = np.load(main_dir + "test_t_candidate.npy", mmap_mode="r")


def load_feat_and_score(mode):
    #mode = "valid"
    feat_dir = "feature_output/%s%s_feats/" % (postfix, mode)
    rrt_feat = np.load(feat_dir + "rrt_feat.npy", mmap_mode="r")
    h2t_t2h_feat = np.load(feat_dir + "h2t_t2h_feat.npy", mmap_mode="r")
    t2h_h2t_feat = np.load(feat_dir + "t2h_h2t_feat.npy", mmap_mode="r")
    h2t_h2t_feat = np.load(feat_dir + "h2t_h2t_feat.npy", mmap_mode="r")
    hht_feat_byprob = np.load(feat_dir + "hht_feat.npy", mmap_mode="r")
    r2t_h2r_feat = np.load(feat_dir + "r2t_h2r_feat.npy", mmap_mode="r")
    r2t_feat = np.load(feat_dir + "r2t_feat.npy", mmap_mode="r")
    rrh_feat = np.load(feat_dir + "rrh_feat.npy", mmap_mode="r")
    rt_count = np.load(feat_dir + "rt_feat.npy", mmap_mode="r")
    ht_count = np.load(feat_dir + "ht_feat.npy", mmap_mode="r")
    feats = [
        rrt_feat, h2t_t2h_feat, t2h_h2t_feat, h2t_h2t_feat, hht_feat_byprob,
        r2t_h2r_feat, r2t_feat, rrh_feat, rt_count, ht_count
    ]
    # model scores
    score_dir = "model_output/%s" % postfix
    ote_names = [
        "ote40/OTE_wikikg90m_concat_d_200_g_12.00",
        "ote20_hd240/OTE_wikikg90m_concat_d_200_g_12.00",
        "ote20_lr0.3/OTE_wikikg90m_concat_d_200_g_12.00",
        "ote20_lrd2w/OTE_wikikg90m_concat_d_200_g_12.00",
        "ote20_lrd5k/OTE_wikikg90m_concat_d_200_g_12.00",
        "ote20_mlplr4e-5/OTE_wikikg90m_concat_d_200_g_12.00",
        "ote20_bs1.2k/OTE_wikikg90m_concat_d_200_g_12.00",
        "ote20_gamma10/OTE_wikikg90m_concat_d_200_g_10.00",
        "ote20_gamma14/OTE_wikikg90m_concat_d_200_g_14.00",
        "ote20/OTE_wikikg90m_concat_d_200_g_12.00",
    ]
    ote_scores = [
        np.load(
            score_dir + name + "/%s_scores.npy" % mode, mmap_mode="r")
        for name in ote_names
    ]
    transe_score = np.load(
        score_dir +
        "/TransE/TransE_l2_wikikg90m_concat_d_200_g_8.00/%s_scores.npy" % mode,
        mmap_mode="r")
    transe_ps_score = np.load(
        score_dir + "/TransE/PostSmoothing/%s_scores.npy" % mode,
        mmap_mode="r")
    rotate_score = np.load(
        score_dir +
        "/RotatE/RotatE_wikikg90m_concat_d_100_g_8.00/%s_scores.npy" % mode,
        mmap_mode="r")
    rotate_ps_score = np.load(
        score_dir + "/RotatE/PostSmoothing/%s_scores.npy" % mode,
        mmap_mode="r")
    quate_score = np.load(
        score_dir + "/QuatE/QuatE_wikikg90m_concat_d_200_g_8.00/%s_scores.npy"
        % mode,
        mmap_mode="r")
    deepwalk_score = np.load(
        score_dir + "/Deepwalk/Deepwalk/%s_scores.npy" % mode, mmap_mode="r")
    scores = ote_scores + [
        transe_score, transe_ps_score, rotate_score, rotate_ps_score,
        quate_score, deepwalk_score
    ]
    return feats, scores


valid_feats, valid_scores = load_feat_and_score("valid")
test_feats, test_scores = load_feat_and_score("test")


def get_mrr(score, idx):
    top10 = np.argsort(-score, -1)[:, :10]
    mrrs = []
    failed = []
    for en, (i, t) in enumerate(zip(idx, top10)):
        if i in t:
            mrr = 1 / (list(t).index(i) + 1)
        else:
            mrr = 0
        if mrr != 1:
            failed.append(en)
        mrrs.append(mrr)
    return sum(mrrs) / len(mrrs)


def ensemble(scores, feats, label=None, num=5000):
    score = 0
    for feat, weight in scores:
        cur_feat = feat[:num].astype(np.float32)
        min_value = cur_feat.min()
        max_value = cur_feat.max()
        cur_feat = (cur_feat - min_value) / (max_value - min_value)
        score += cur_feat * weight
    score = score / len(scores)

    for feat, weight in feats:
        cur_feat = feat[:num].astype(np.float32)
        score += cur_feat * weight
    if label is None:
        return score
    label = label[:num]
    return get_mrr(score, label), score


scores = [
    (test_scores[0], 1),
    (test_scores[1], 1),
    (test_scores[2], 1),
    (test_scores[3], 1),
    (test_scores[4], 1),
    (test_scores[5], 1),
    (test_scores[6], 1),
    (test_scores[7], 1),
    (test_scores[8], 1),
    (test_scores[9], 1),
    (test_scores[10], 1),
    (test_scores[11], 1),
    (test_scores[12], 1),
    (test_scores[13], 1),
    (test_scores[14], 1),
    (test_scores[15], 1),
]
feats = [
    (test_scores[0], 1000),
    (test_scores[1], 300),
    (test_scores[2], 60),
    (test_scores[3], 10),
    (test_scores[4], 3),
    (test_scores[5], 1000),
    (test_scores[6], 0.01),
    (test_scores[7], 0.01),
    (test_scores[8], 0.01),
    (test_scores[9], 0.1),
]

score = ensemble(scores, feats, None, 500000000000)

top10 = np.argsort(-score, -1)[:, :10]
evaluator = WikiKG90MEvaluator()
best_test_dict = {"h,r->t": {"t_pred_top10": top10, }}
path = "./"
evaluator.save_test_submission(best_test_dict, path)
