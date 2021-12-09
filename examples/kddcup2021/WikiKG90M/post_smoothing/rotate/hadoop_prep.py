# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import sys
sys.path.append(".")

import yaml
from easydict import EasyDict as edict
import argparse
import numpy as np
import json
import six

if six.PY3:
    import io
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding='utf-8', errors='ignore')
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding='utf-8', errors='ignore')


def compact_json(out_dict):
    return json.dumps(out_dict, separators=(",", ":"), ensure_ascii=False)


def get_indegree_mapper(conf):
    """
    Input:
        src0 \t rela0 \t dst0
        src1 \t rela1 \t dst1
        src2 \t rela2 \t dst2
        src0 \t nfeat

    Output:
        dst0 \t 0 \t nfeat
        dst0 \t 1 \t src0
        dst0 \t 1 \t src1
        dst0 \t 1 \t src2
    """
    for line in sys.stdin:
        #src, dst_or_nfeat = line.strip().split('\t')
        slots = line.strip().split('\t')
        #if len(dst_or_nfeat.split(' ')) > 2:  # nfeat
        if len(slots) == 2:
            src, dst_or_nfeat = slots[0], slots[1]
            sys.stdout.write("%s\t0\t%s\n" % (src, dst_or_nfeat))
        else:
            src, dst_or_nfeat = slots[0], slots[2]
            sys.stdout.write("%s\t1\t%s\n" % (dst_or_nfeat, src))
            if conf.bidirect:
                sys.stdout.write("%s\t1\t%s\n" % (src, dst_or_nfeat))


def get_indegree_reducer(conf):
    """
    get indegree of every node

    Input:
        dst0 \t 0 \t nfeat
        dst0 \t 1 \t src0
        dst0 \t 1 \t src1
        dst0 \t 1 \t src2

    Output:
        dst0 \t indegree0 ; nfeat0
        dst1 \t indegree1 ; nfeat1
    """
    cur_dst = None
    cur_nfeat = None
    indegree = 0  # for self loop
    for line in sys.stdin:
        nid, mode, nid_or_nfeat = line.strip().split('\t')
        mode = int(mode)
        if mode == 0:
            if cur_dst is not None:
                if indegree == 0:
                    indegree = 1
                sys.stdout.write("%s\t%s;%s\n" %
                                 (cur_dst, indegree, cur_nfeat))

            cur_dst = nid
            cur_nfeat = nid_or_nfeat
            indegree = 0

        else:
            indegree += 1

    if indegree == 0:
        indegree = 1
    sys.stdout.write("%s\t%s;%s\n" % (cur_dst, indegree, cur_nfeat))


def merge_edges_and_nfeat(conf, runs):
    """
    Input:
        nid0 \t indegree0 ; nfeat0
        nid1 \t indegree1 ; nfeat1
        src0 \t rela0 \t dst0
        src1 \t rela1 \t dst1
        
    return:
        src \t 0 \t indegree ; nfeat
        src \t runs \t dst0
        src \t runs \t dst1
        src \t runs \t dst2
    """
    for line in sys.stdin:
        fields = line.strip().split('\t')
        if len(fields) == 2:
            # it is node feature
            nid, indegree_nfeat = fields
            sys.stdout.write("%s\t%s\t%s\n" % (nid, 0, indegree_nfeat))
        else:
            #src, dst = fields
            src, r_type, dst = fields[0], fields[1], fields[2]
            # reverse for join
            sys.stdout.write("%s\t%s\t%s\t%s\t%s\n" %
                             (src, runs, dst, r_type, 0))

            if conf.bidirect:
                dst, src = src, dst
                # reverse for join
                sys.stdout.write("%s\t%s\t%s\t%s\t%s\n" %
                                 (src, runs, dst, r_type, 1))


def join_src(conf):
    """
    Input:
        src \t 0 \t indegree ; nfeat
        src \t runs \t dst0 \t rela0 \t direct0
        src \t runs \t dst1 \t rela1 \t direct1
        src \t runs \t dst2 \t rela2 \t direct2

    Output:
        src \t 0 \t indegree ; nfeat
        dst0 \t runs \t indegree ; src0_nfeat \t rela0 \t direct0
        dst1 \t runs \t indegree ; src1_nfeat \t rela1 \t direct1
        
    """
    src_feat = None
    last_src = None
    for line in sys.stdin:
        #src, mode, dst_or_src_feat = line.strip('\r\n').split("\t")
        fields = line.strip('\r\n').split("\t")
        src, mode = fields[0], fields[1]
        if last_src != src:
            src_feat = None
        last_src = src

        if mode != "0":
            if src_feat is not None:
                #sys.stdout.write("%s\t%s\t%s\n" % (dst_or_src_feat, mode, src_feat))
                dst, r_type, direct = fields[2], fields[3], fields[4]
                sys.stdout.write("%s\t%s\t%s\t%s\t%s\n" %
                                 (dst, mode, src_feat, r_type, direct))

        elif mode == "0":
            src_feat = fields[2]
            sys.stdout.write(line.strip("\r\n") + '\n')


def reduce_neigh(conf, runs):
    """
    Input:
        dst(nid) \t 0 \t dst_indegree ; dst_nfeat
        dst0 \t runs \t src_indegree ; src_nfeat \t rela0 \t direct0
        dst0 \t runs \t src_indegree ; src_nfeat \t rela1 \t direct1
        dst0 \t runs \t src_indegree ; src_nfeat \t rela2 \t direct2

    Output:
        dst0 \t dst_indegree ; new_feat
        dst1 \t dst_indegree ; new_feat
    """
    islast = conf.khop == runs  # if True, means the last hop
    cur_dst = None
    cur_dst_nfeat = None
    neigh_nfeat_list = []
    r_emb = np.load("wikikg90m_TransE_l2_relation.npy")
    last_dst = None
    for line in sys.stdin:
        #dst, mode, src_feat_or_dst_feat = line.strip().split('\t')
        fields = line.strip().split('\t')
        dst, mode, src_feat_or_dst_feat = fields[0], fields[1], fields[2]

        if last_dst is not None and last_dst != dst:
            new_nfeat = ppnp_aggr(cur_dst_nfeat, neigh_nfeat_list, conf.alpha,
                                  islast, r_emb)
            sys.stdout.write("%s\t%s\n" % (cur_dst, new_nfeat))
            neigh_nfeat_list = []

        mode = int(mode)
        if mode == 0:
            cur_dst = dst
            cur_dst_nfeat = src_feat_or_dst_feat
        else:
            if len(neigh_nfeat_list) < 50000:
                r_type = fields[3]
                direct = fields[4]
                neigh_nfeat_list.append([src_feat_or_dst_feat, r_type, direct])

        last_dst = dst

        # mode = int(mode)
        # if mode == 0:
    # if cur_dst is not None and cur_dst_nfeat is not None: # not the first line
    # new_nfeat = ppnp_aggr(cur_dst_nfeat, neigh_nfeat_list, conf.alpha, islast, r_emb)
    # sys.stdout.write("%s\t%s\n" % (cur_dst, new_nfeat))

    # cur_dst = dst
    # cur_dst_nfeat = src_feat_or_dst_feat
    # neigh_nfeat_list = []

    # else:
    # r_type = fields[3]
    # direct = fields[4]
    # neigh_nfeat_list.append([src_feat_or_dst_feat, r_type, direct])

    if cur_dst_nfeat is not None:
        new_nfeat = ppnp_aggr(cur_dst_nfeat, neigh_nfeat_list, conf.alpha,
                              islast, r_emb)
        sys.stdout.write("%s\t%s\n" % (cur_dst, new_nfeat))


def ppnp_aggr(dst_nfeat, neigh_nfeat_list, alpha=0.2, islast=False,
              r_emb=None):
    dst_indegree, dst_nfeat = dst_nfeat.split(";")
    dst_norm = np.power(int(dst_indegree), -0.5)
    dst_nfeat = [float(item) for item in dst_nfeat.split(' ')]
    dst_nfeat = np.array(dst_nfeat, dtype="float32").reshape(1, -1)

    tmp_nfeat_list = []
    tmp_src_indegree = []
    for line in neigh_nfeat_list:
        src_feat_or_dst_feat, r_type, direct = line
        indegree, src_nfeat = src_feat_or_dst_feat.split(";")
        tmp_src_indegree.append(int(indegree))
        #tmp_nfeat_list.append([float(item) for item in src_nfeat.split(' ')])
        src_nfeat_value = [float(item) for item in src_nfeat.split(' ')]
        if 0:
            if direct == "1":
                sign = -1
            elif direct == "0":
                sign = 1
            else:
                raise ValueError
            src_nfeat_value = np.array(src_nfeat_value) + r_emb[int(
                r_type)] * sign
        else:
            gamma = 10
            hidden_dim = 100
            emb_init = gamma / hidden_dim
            relation = r_emb[int(r_type)]
            phase_rel = relation / (emb_init / np.pi)
            re_rel, im_rel = np.cos(phase_rel), np.sin(phase_rel)
            head = np.array(src_nfeat_value, dtype=np.float16)
            if direct == "1":  # h = t - r
                re_head, im_head = np.split(head, 2, -1)
                re_score = re_head * re_rel + im_head * im_rel
                im_score = -re_head * im_rel + im_head * re_rel
                src_nfeat_value = np.concatenate([re_score, im_score], -1)
            elif direct == "0":  # t = h + r 
                re_head, im_head = np.split(head, 2, -1)
                re_score = re_head * re_rel - im_head * im_rel
                im_score = re_head * im_rel + im_head * re_rel
                src_nfeat_value = np.concatenate([re_score, im_score], -1)
            else:
                raise ValueError
            #src_nfeat_value = np.array(src_nfeat_value) + r_emb[int(r_type)] * sign
        tmp_nfeat_list.append(src_nfeat_value)

    if len(tmp_nfeat_list) > 0:
        neigh_nfeat = np.array(tmp_nfeat_list, dtype="float32")
        aggr_nfeat = np.mean(neigh_nfeat, axis=0)
        new_nfeat = dst_nfeat * alpha + aggr_nfeat * (1 - alpha)
    else:
        new_nfeat = dst_nfeat

    new_nfeat = new_nfeat.reshape(-1, )
    new_nfeat = ["%.7f" % i for i in new_nfeat]
    new_nfeat = ' '.join(new_nfeat)

    if not islast:
        new_nfeat = dst_indegree + ';' + new_nfeat
    return new_nfeat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--conf", type=str, default="./smoothing_config.yaml")
    parser.add_argument("-m", "--mode", type=str, default="mapper")
    parser.add_argument("-r", "--runs", type=int, default=0)
    args = parser.parse_args()
    config = edict(yaml.load(open(args.conf), Loader=yaml.FullLoader))
    if args.mode == "merge":
        merge_edges_and_nfeat(config, args.runs)
    elif args.mode == "join_src":
        join_src(config)
    elif args.mode == "reduce_neigh":
        reduce_neigh(config, args.runs)
    elif args.mode == "indegree_map":
        get_indegree_mapper(config)
    elif args.mode == "indegree_reduce":
        get_indegree_reducer(config)
