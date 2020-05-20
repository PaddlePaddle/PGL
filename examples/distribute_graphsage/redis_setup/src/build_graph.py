# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import json
import logging
from collections import defaultdict
import tqdm
import redis
from redis._compat import b, unicode, bytes, long, basestring
from rediscluster.nodemanager import NodeManager
from rediscluster.crc import crc16
import argparse
import time
import pickle
import numpy as np
import scipy.sparse as sp

log = logging.getLogger(__name__)
root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)


def encode(value):
    """
    Return a bytestring representation of the value.
    This method is copied from Redis' connection.py:Connection.encode
    """
    if isinstance(value, bytes):
        return value
    elif isinstance(value, (int, long)):
        value = b(str(value))
    elif isinstance(value, float):
        value = b(repr(value))
    elif not isinstance(value, basestring):
        value = unicode(value)
    if isinstance(value, unicode):
        value = value.encode('utf-8')
    return value


def crc16_hash(data):
    return crc16(encode(data))


def get_redis(startup_host, startup_port):
    startup_nodes = [{"host": startup_host, "port": startup_port}, ]
    nodemanager = NodeManager(startup_nodes=startup_nodes)
    nodemanager.initialize()
    rs = {}
    for node, config in nodemanager.nodes.items():
        rs[node] = redis.Redis(
            host=config["host"], port=config["port"], decode_responses=False)
    return rs, nodemanager


def load_data(edge_path):
    src, dst = [], []
    with open(edge_path, "r") as f:
        for i in tqdm.tqdm(f):
            s, d, _ = i.split()
            s = int(s)
            d = int(d)
            src.append(s)
            dst.append(d)
            dst.append(s)
            src.append(d)
    src = np.array(src, dtype="int64")
    dst = np.array(dst, dtype="int64")
    return src, dst


def build_edge_index(edge_path, num_nodes, startup_host, startup_port,
                     num_bucket):
    #src, dst = load_data(edge_path)
    rs, nodemanager = get_redis(startup_host, startup_port)

    dst_mp, edge_mp = defaultdict(list), defaultdict(list)
    with open(edge_path) as f:
        for l in tqdm.tqdm(f):
            a, b, idx = l.rstrip().split('\t')
            a, b, idx = int(a), int(b), int(idx)
            dst_mp[a].append(b)
            edge_mp[a].append(idx)
    part_dst_dicts = {}
    for i in tqdm.tqdm(range(num_nodes)):
        #if len(edge_index.v[i]) == 0:
        #    continue
        #v = edge_index.v[i].astype("int64").reshape([-1, 1])
        #e = edge_index.eid[i].astype("int64").reshape([-1, 1])
        if i not in dst_mp:
            continue
        v = np.array(dst_mp[i]).astype('int64').reshape([-1, 1])
        e = np.array(edge_mp[i]).astype('int64').reshape([-1, 1])
        o = np.hstack([v, e])
        key = "d:%s" % i
        part = crc16_hash(key) % num_bucket
        if part not in part_dst_dicts:
            part_dst_dicts[part] = {}
        dst_dicts = part_dst_dicts[part]
        dst_dicts["d:%s" % i] = o.tobytes()
        if len(dst_dicts) > 10000:
            slot = nodemanager.keyslot("part-%s" % part)
            node = nodemanager.slots[slot][0]['name']
            while True:
                res = rs[node].hmset("part-%s" % part, dst_dicts)
                if res:
                    break
                log.info("HMSET FAILED RETRY connected %s" % node)
                time.sleep(1)
            part_dst_dicts[part] = {}

    for part, dst_dicts in part_dst_dicts.items():
        if len(dst_dicts) > 0:
            slot = nodemanager.keyslot("part-%s" % part)
            node = nodemanager.slots[slot][0]['name']
            while True:
                res = rs[node].hmset("part-%s" % part, dst_dicts)
                if res:
                    break
                log.info("HMSET FAILED RETRY connected %s" % node)
                time.sleep(1)
            part_dst_dicts[part] = {}
    log.info("dst_dict Done")


def build_edge_id(edge_path, num_nodes, startup_host, startup_port,
                  num_bucket):
    src, dst = load_data(edge_path)
    rs, nodemanager = get_redis(startup_host, startup_port)
    part_edge_dict = {}
    for i in tqdm.tqdm(range(len(src))):
        key = "e:%s" % i
        part = crc16_hash(key) % num_bucket
        if part not in part_edge_dict:
            part_edge_dict[part] = {}
        edge_dict = part_edge_dict[part]
        edge_dict["e:%s" % i] = int(src[i]) * num_nodes + int(dst[i])
        if len(edge_dict) > 10000:
            slot = nodemanager.keyslot("part-%s" % part)
            node = nodemanager.slots[slot][0]['name']
            while True:
                res = rs[node].hmset("part-%s" % part, edge_dict)
                if res:
                    break
                log.info("HMSET FAILED RETRY connected %s" % node)
                time.sleep(1)

            part_edge_dict[part] = {}

    for part, edge_dict in part_edge_dict.items():
        if len(edge_dict) > 0:
            slot = nodemanager.keyslot("part-%s" % part)
            node = nodemanager.slots[slot][0]['name']
            while True:
                res = rs[node].hmset("part-%s" % part, edge_dict)
                if res:
                    break
                log.info("HMSET FAILED RETRY connected %s" % node)
                time.sleep(1)
            part_edge_dict[part] = {}


def build_infos(edge_path, num_nodes, startup_host, startup_port, num_bucket):
    src, dst = load_data(edge_path)
    rs, nodemanager = get_redis(startup_host, startup_port)
    slot = nodemanager.keyslot("num_nodes")
    node = nodemanager.slots[slot][0]['name']
    res = rs[node].set("num_nodes", num_nodes)

    slot = nodemanager.keyslot("num_edges")
    node = nodemanager.slots[slot][0]['name']
    rs[node].set("num_edges", len(src))

    slot = nodemanager.keyslot("nf:infos")
    node = nodemanager.slots[slot][0]['name']
    rs[node].set("nf:infos", json.dumps([['feats', [-1, 602], 'float32'], ]))

    slot = nodemanager.keyslot("ef:infos")
    node = nodemanager.slots[slot][0]['name']
    rs[node].set("ef:infos", json.dumps([]))


def build_node_feat(node_feat_path, num_nodes, startup_host, startup_port, num_bucket):
    assert node_feat_path != "", "node_feat_path empty!"
    feat_dict = np.load(node_feat_path)
    for k in feat_dict.keys():
        feat = feat_dict[k]
        assert feat.shape[0] == num_nodes, "num_nodes invalid"

    rs, nodemanager = get_redis(startup_host, startup_port)
    part_feat_dict = {}
    for k in feat_dict.keys():
        feat = feat_dict[k]
        for i in tqdm.tqdm(range(num_nodes)):
            key = "nf:%s:%i" % (k, i)
            value = feat[i].tobytes()
            part = crc16_hash(key) % num_bucket
            if part not in part_feat_dict:
                part_feat_dict[part] = {}
            part_feat = part_feat_dict[part]
            part_feat[key] = value
            if len(part_feat) > 100:
                slot = nodemanager.keyslot("part-%s" % part)
                node = nodemanager.slots[slot][0]['name']
                while True:
                    res = rs[node].hmset("part-%s" % part, part_feat)
                    if res:
                        break
                    log.info("HMSET FAILED RETRY connected %s" % node)
                    time.sleep(1)

                part_feat_dict[part] = {}

    for part, part_feat in part_feat_dict.items():
        if len(part_feat) > 0:
            slot = nodemanager.keyslot("part-%s" % part)
            node = nodemanager.slots[slot][0]['name']
            while True:
                res = rs[node].hmset("part-%s" % part, part_feat)
                if res:
                    break
                log.info("HMSET FAILED RETRY connected %s" % node)
                time.sleep(1)
            part_feat_dict[part] = {}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gen_redis_conf')
    parser.add_argument('--startup_port', type=int, required=True)
    parser.add_argument('--startup_host', type=str, required=True)
    parser.add_argument('--edge_path', type=str, default="")
    parser.add_argument('--node_feat_path', type=str, default="")
    parser.add_argument('--num_nodes', type=int, default=0)
    parser.add_argument('--num_bucket', type=int, default=64)
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        help="choose one of the following modes (clear, edge_index, edge_id, graph_attr)"
    )
    args = parser.parse_args()
    log.info("Mode: {}".format(args.mode))
    if args.mode == 'edge_index':
        build_edge_index(args.edge_path, args.num_nodes, args.startup_host,
                         args.startup_port, args.num_bucket)
    elif args.mode == 'edge_id':
        build_edge_id(args.edge_path, args.num_nodes, args.startup_host,
                      args.startup_port, args.num_bucket)
    elif args.mode == 'graph_attr':
        build_infos(args.edge_path, args.num_nodes, args.startup_host,
                    args.startup_port, args.num_bucket)
    elif args.mode == 'node_feat':
        build_node_feat(args.node_feat_path, args.num_nodes, args.startup_host,
                    args.startup_port, args.num_bucket)
    else:
        raise ValueError("%s mode not found" % args.mode)

