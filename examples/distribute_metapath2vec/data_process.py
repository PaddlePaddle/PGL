# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
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
"""Data preprocessing for DBLP dataset"""
import sys
import os
import argparse
import numpy as np
from collections import OrderedDict

AUTHOR = 14475
PAPER = 14376
CONF = 20
TYPE = 8920
LABEL = 4


def build_node_types(meta_node, outfile):
    """build_node_types"""
    nt_ori2new = {}
    with open(outfile, 'w') as writer:
        offset = 0
        for node_type, num_nodes in meta_node.items():
            ori_id2new_id = {}
            for i in range(num_nodes):
                writer.write("%d\t%s\n" % (offset + i, node_type))
                ori_id2new_id[i + 1] = offset + i
            nt_ori2new[node_type] = ori_id2new_id
            offset += num_nodes
    return nt_ori2new


def remapping_index(args, src_dict, dst_dict, ori_file, new_file):
    """remapping_index"""
    ori_file = os.path.join(args.data_path, ori_file)
    new_file = os.path.join(args.output_path, new_file)
    with open(ori_file, 'r') as reader, open(new_file, 'w') as writer:
        for line in reader:
            slots = line.strip().split()
            s = int(slots[0])
            d = int(slots[1])
            new_s = src_dict[s]
            new_d = dst_dict[d]
            writer.write("%d\t%d\n" % (new_s, new_d))


def author_label(args, ori_id2pgl_id, ori_file, real_file, new_file):
    """author_label"""
    ori_file = os.path.join(args.data_path, ori_file)
    real_file = os.path.join(args.data_path, real_file)
    new_file = os.path.join(args.output_path, new_file)
    real_id2pgl_id = {}
    with open(ori_file, 'r') as reader:
        for line in reader:
            slots = line.strip().split()
            ori_id = int(slots[0])
            real_id = int(slots[1])
            pgl_id = ori_id2pgl_id[ori_id]
            real_id2pgl_id[real_id] = pgl_id

    with open(real_file, 'r') as reader, open(new_file, 'w') as writer:
        for line in reader:
            slots = line.strip().split()
            real_id = int(slots[0])
            label = int(slots[1])
            pgl_id = real_id2pgl_id[real_id]
            writer.write("%d\t%d\n" % (pgl_id, label))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DBLP data preprocessing')
    parser.add_argument(
        '--data_path',
        default=None,
        type=str,
        help='original data path(default: None)')
    parser.add_argument(
        '--output_path',
        default=None,
        type=str,
        help='output path(default: None)')
    args = parser.parse_args()

    meta_node = OrderedDict()
    meta_node['a'] = AUTHOR
    meta_node['p'] = PAPER
    meta_node['c'] = CONF
    meta_node['t'] = TYPE

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    node_types_file = os.path.join(args.output_path, "node_types.txt")
    nt_ori2new = build_node_types(meta_node, node_types_file)

    remapping_index(args, nt_ori2new['p'], nt_ori2new['a'], 'paper_author.dat',
                    'paper_author.txt')
    remapping_index(args, nt_ori2new['p'], nt_ori2new['c'],
                    'paper_conference.dat', 'paper_conference.txt')
    remapping_index(args, nt_ori2new['p'], nt_ori2new['t'], 'paper_type.dat',
                    'paper_type.txt')

    author_label(args, nt_ori2new['a'], 'author_map_id.dat',
                 'author_label.dat', 'author_label.txt')
