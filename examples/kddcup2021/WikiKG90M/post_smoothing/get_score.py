#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
#
# File: valid_score.py
# Author: suweiyue(suweiyue@baidu.com)
# Date: 2021/04/19 14:09:25
#
########################################################################
"""
    Comment.
"""
import numpy as np
from glob import glob
from tqdm import tqdm

import sys
import argparse
import logging
from multiprocessing import Pool

log = logging.getLogger(__name__)
stream_hdl = logging.StreamHandler(stream=sys.stderr)
formatter = logging.Formatter(
    fmt='[%(levelname)s] %(asctime)s [%(filename)12s:%(lineno)5d]:	%(message)s')
stream_hdl.setFormatter(formatter)
log.addHandler(stream_hdl)


def m2vscore(head, relation, tails):
    return -(head * tails).sum(-1)


def transEscore(head, relation, tails):
    score = np.linalg.norm(head + relation - tails, 2, -1)  # B,1001
    return score


def rotatEscore(head, relation, tails):
    re_head, im_head = np.split(head, 2, -1)
    re_tail, im_tail = np.split(tails, 2, -1)
    gamma = 10
    hidden_dim = 100
    #gamma = 8
    #hidden_dim = 200
    emb_init = gamma / hidden_dim
    phase_rel = relation / (emb_init / np.pi)
    #phase_rel = relation
    re_rel, im_rel = np.cos(phase_rel), np.sin(phase_rel)
    re_score = re_head * re_rel - im_head * im_rel
    im_score = re_head * im_rel + im_head * re_rel
    re_score = re_score - re_tail  # [B, 1001, 768]
    im_score = im_score - im_tail  # [B, 1001, 768]
    score = np.stack([re_score, im_score], 0)
    #score = score.norm(dim=0)
    score = np.linalg.norm(score, 2, 0)  # B,1001
    score = score.sum(-1)
    return score


def func(start):
    #print("start")
    if args.mode == "valid":
        val_hr = np.load(
            "./dataset/wikikg90m_kddcup2021/processed/val_hr.npy",
            mmap_mode="r")
        val_t_candidate = np.load(
            "./dataset/wikikg90m_kddcup2021/processed/val_t_candidate.npy",
            mmap_mode="r")
    else:
        val_hr = np.load(
            "./dataset/wikikg90m_kddcup2021/processed/test_hr.npy",
            mmap_mode="r")
        val_t_candidate = np.load(
            "./dataset/wikikg90m_kddcup2021/processed/test_t_candidate.npy",
            mmap_mode="r")

    if args.relation_path is not None:
        relatioin_emb = np.load(args.relation_path, mmap_mode="r")
    entity_emb = np.load(args.tmp_path, mmap_mode="r")

    buf_size = 1000
    end = start + buf_size
    head = entity_emb[val_hr[start:end, 0:1]]  # B,1,H
    relation = relatioin_emb[val_hr[start:end, 1:2]]  # B,1,H
    tails = entity_emb[val_t_candidate[start:end]]  # B,1001,H

    if args.score_func == "TransE":
        score = transEscore(head, relation, tails)
    elif args.score_func == "RotatE":
        score = rotatEscore(head, relation, tails)
    elif args.score_func == "m2v":
        score = m2vscore(head, relation, tails)
    else:
        raise ValueError

    return score


def get_top10(val_hr):
    buf_size = 1000
    rets = []
    args_arr = [start for start in range(0, val_hr.shape[0], buf_size)]
    with Pool(20) as p:
        rets = p.map(func, args_arr)
    return np.concatenate(rets, 0)


def main(args):

    if args.mode == "valid":
        val_hr = np.load(
            "./dataset/wikikg90m_kddcup2021/processed/val_hr.npy",
            mmap_mode="r")[:args.nums]
    else:
        val_hr = np.load(
            "./dataset/wikikg90m_kddcup2021/processed/test_hr.npy",
            mmap_mode="r")[:args.nums]
    print(args.tmp_path)

    scores = get_top10(val_hr)
    if args.output_path is not None:
        np.save(args.output_path, scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--tmp_path", type=str, default="entity_emb.npy")
    parser.add_argument("--relation_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--score_func", type=str, default="TransE")
    parser.add_argument("--mode", type=str, default="valid")
    parser.add_argument("--nums", type=int, default=20000000)
    args = parser.parse_args()
    main(args)
