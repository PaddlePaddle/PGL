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
"""redis_graph"""

import pgl
import redis
from redis import BlockingConnectionPool, StrictRedis
from redis._compat import b, unicode, bytes, long, basestring
from rediscluster.nodemanager import NodeManager
from rediscluster.crc import crc16
from collections import OrderedDict
import threading
import numpy as np
import time
import json
import pgl.graph as pgraph
import pickle as pkl
from pgl.utils.logger import log
import pgl.graph_kernel as graph_kernel


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
    """crc16_hash"""
    return crc16(encode(data))


LUA_SCRIPT = """
math.randomseed(tonumber(ARGV[1]))

local function permute(tab, count, bucket_size)
    local n = #tab / bucket_size
    local o_ret = {}
    local o_dict = {}
    for i = 1, count do
        local j = math.random(i, n)
        o_ret[i] = string.sub(tab, (i - 1) * bucket_size + 1, i * bucket_size)
        if j > count then
            if o_dict[j] ~= nil then
                o_ret[i], o_dict[j] = o_dict[j], o_ret[i]
            else
                o_dict[j], o_ret[i] = o_ret[i], string.sub(tab, (j - 1) * bucket_size + 1, j * bucket_size)
            end
        end
    end
    return table.concat(o_ret)
end

local bucket_size = 16
local ret = {}
local sample_size = tonumber(ARGV[2])
for i=1, #ARGV - 2 do
    local tab = redis.call("HGET", KEYS[1], ARGV[i + 2])
    if tab then
        if #tab / bucket_size <= sample_size then
            ret[i] = tab
        else
            ret[i] = permute(tab, sample_size, bucket_size)
        end
    else
        ret[i] = tab
    end
end
return ret
"""


class RedisCluster(object):
    """RedisCluster"""

    def __init__(self, startup_nodes):
        self.nodemanager = NodeManager(startup_nodes=startup_nodes)
        self.nodemanager.initialize()
        self.redis_worker = {}
        for node, config in self.nodemanager.nodes.items():
            rdp = BlockingConnectionPool(
                host=config["host"], port=config["port"])
            self.redis_worker[node] = {
                "worker": StrictRedis(
                    connection_pool=rdp, decode_responses=False),
                "type": config["server_type"]
            }

    def get(self, key):
        """get"""
        slot = self.nodemanager.keyslot(key)
        node = np.random.choice(self.nodemanager.slots[slot])
        worker = self.redis_worker[node['name']]
        if worker["type"] == "slave":
            worker["worker"].execute_command("READONLY")
        return worker["worker"].get(key)

    def hmget(self, key, fields):
        """hmget"""
        while True:
            retry = 0
            try:
                slot = self.nodemanager.keyslot(key)
                node = np.random.choice(self.nodemanager.slots[slot])
                worker = self.redis_worker[node['name']]
                if worker["type"] == "slave":
                    worker["worker"].execute_command("READONLY")
                ret = worker["worker"].hmget(key, fields)
                break
            except Exception as e:
                retry += 1
                if retry > 5:
                    raise e
                print("RETRY  hmget after 1 sec. Retry Time %s" % retry)
                time.sleep(1)
        return ret

    def hmget_sample(self, key, fields, sample):
        """hmget_sample"""
        while True:
            retry = 0
            try:
                slot = self.nodemanager.keyslot(key)
                node = np.random.choice(self.nodemanager.slots[slot])
                worker = self.redis_worker[node['name']]
                if worker["type"] == "slave":
                    worker["worker"].execute_command("READONLY")
                func = worker["worker"].register_script(LUA_SCRIPT)
                ret = func(
                    keys=[key],
                    args=[np.random.randint(4294967295), sample] + fields)
                break
            except Exception as e:
                retry += 1
                if retry > 5:
                    raise e
                print("RETRY  hmget_sample after 1 sec. Retry Time %s" % retry)
                time.sleep(1)
        return ret


def hmget_sample_helper(rs, query, num_parts, sample_size):
    """hmget_sample_helper"""
    buff = [b""] * len(query)
    part_dict = {}
    part_ind_dict = {}
    for ind, q in enumerate(query):
        part = crc16_hash(q) % num_parts
        part = "part-%s" % part
        if part not in part_dict:
            part_dict[part] = []
            part_ind_dict[part] = []
        part_dict[part].append(q)
        part_ind_dict[part].append(ind)

    def worker(_key, _value, _buff, _rs, _part_ind_dict, _sample_size):
        """worker"""
        response = _rs.hmget_sample(_key, _value, _sample_size)
        for res, ind in zip(response, _part_ind_dict[_key]):
            buff[ind] = res

    def hmget(_part_dict, _rs, _buff, _part_ind_dict, _sample_size):
        """hmget"""
        key_value = list(_part_dict.items())
        np.random.shuffle(key_value)
        for key, value in key_value:
            worker(key, value, _buff, _rs, _part_ind_dict, _sample_size)

    hmget(part_dict, rs, buff, part_ind_dict, sample_size)
    return buff


def hmget_helper(rs, query, num_parts):
    """hmget_helper"""
    buff = [b""] * len(query)
    part_dict = {}
    part_ind_dict = {}
    for ind, q in enumerate(query):
        part = crc16_hash(q) % num_parts
        part = "part-%s" % part
        if part not in part_dict:
            part_dict[part] = []
            part_ind_dict[part] = []
        part_dict[part].append(q)
        part_ind_dict[part].append(ind)

    def worker(_key, _value, _buff, _rs, _part_ind_dict):
        """worker"""
        response = _rs.hmget(_key, _value)
        for res, ind in zip(response, _part_ind_dict[_key]):
            buff[ind] = res

    def hmget(_part_dict, _rs, _buff, _part_ind_dict):
        """hmget"""
        key_value = list(_part_dict.items())
        np.random.shuffle(key_value)
        for key, value in key_value:
            worker(key, value, _buff, _rs, _part_ind_dict)

    hmget(part_dict, rs, buff, part_ind_dict)
    return buff


class RedisGraph(pgraph.Graph):
    """RedisGraph"""

    def __init__(self, name, redis_config, num_parts):
        self._rs = RedisCluster(startup_nodes=redis_config)
        self.num_parts = num_parts
        self._name = name
        self._num_nodes = None
        self._num_edges = None
        self._node_feat_info = None
        self._edge_feat_info = None
        self._node_feat_dtype = None
        self._edge_feat_dtype = None
        self._node_feat_shape = None
        self._edge_feat_shape = None

    @property
    def num_nodes(self):
        """num_nodes"""
        if self._num_nodes is None:
            self._num_nodes = int(self._rs.get("num_nodes"))
        return self._num_nodes

    @property
    def num_edges(self):
        """num_edges"""
        if self._num_edges is None:
            self._num_edges = int(self._rs.get("num_edges"))
        return self._num_edges

    def node_feat_info(self):
        """node_feat_info"""
        if self._node_feat_info is None:
            buff = self._rs.get("nf:infos")
            self._node_feat_info = json.loads(buff.decode())
        return self._node_feat_info

    def node_feat_dtype(self, key):
        """node_feat_dtype"""
        if self._node_feat_dtype is None:
            self._node_feat_dtype = {}
            for key, _, dtype in self.node_feat_info():
                self._node_feat_dtype[key] = dtype
        return self._node_feat_dtype[key]

    def node_feat_shape(self, key):
        """node_feat_shape"""
        if self._node_feat_shape is None:
            self._node_feat_shape = {}
            for key, shape, _ in self.node_feat_info():
                self._node_feat_shape[key] = shape
        return self._node_feat_shape[key]

    def edge_feat_shape(self, key):
        """edge_feat_shape"""
        if self._edge_feat_shape is None:
            self._edge_feat_shape = {}
            for key, shape, _ in self.edge_feat_info():
                self._edge_feat_shape[key] = shape
        return self._edge_feat_shape[key]

    def edge_feat_dtype(self, key):
        """edge_feat_dtype"""
        if self._edge_feat_dtype is None:
            self._edge_feat_dtype = {}
            for key, _, dtype in self.edge_feat_info():
                self._edge_feat_dtype[key] = dtype
        return self._edge_feat_dtype[key]

    def edge_feat_info(self):
        """edge_feat_info"""
        if self._edge_feat_info is None:
            buff = self._rs.get("ef:infos")
            self._edge_feat_info = json.loads(buff.decode())
        return self._edge_feat_info

    def sample_predecessor(self, nodes, max_degree, return_eids=False):
        """sample_predecessor"""
        query = ["d:%s" % n for n in nodes]
        rets = hmget_sample_helper(self._rs, query, self.num_parts, max_degree)
        v = []
        eid = []
        for buff in rets:
            if buff is None:
                v.append(np.array([], dtype="int64"))
                eid.append(np.array([], dtype="int64"))
            else:
                npret = np.frombuffer(
                    buff, dtype="int64").reshape([-1, 2]).astype("int64")
                v.append(npret[:, 0])
                eid.append(npret[:, 1])
        if return_eids:
            return np.array(v), np.array(eid)
        else:
            return np.array(v)

    def sample_successor(self, nodes, max_degree, return_eids=False):
        """sample_successor"""
        query = ["s:%s" % n for n in nodes]
        rets = hmget_sample_helper(self._rs, query, self.num_parts, max_degree)
        v = []
        eid = []
        for buff in rets:
            if buff is None:
                v.append(np.array([], dtype="int64"))
                eid.append(np.array([], dtype="int64"))
            else:
                npret = np.frombuffer(
                    buff, dtype="int64").reshape([-1, 2]).astype("int64")
                v.append(npret[:, 0])
                eid.append(npret[:, 1])
        if return_eids:
            return np.array(v), np.array(eid)
        else:
            return np.array(v)

    def predecessor(self, nodes, return_eids=False):
        """predecessor"""
        query = ["d:%s" % n for n in nodes]
        ret = hmget_helper(self._rs, query, self.num_parts)
        v = []
        eid = []
        for buff in ret:
            if buff is not None:
                npret = np.frombuffer(
                    buff, dtype="int64").reshape([-1, 2]).astype("int64")
                v.append(npret[:, 0])
                eid.append(npret[:, 1])
            else:
                v.append(np.array([], dtype="int64"))
                eid.append(np.array([], dtype="int64"))
        if return_eids:
            return np.array(v), np.array(eid)
        else:
            return np.array(v)

    def successor(self, nodes, return_eids=False):
        """successor"""
        query = ["s:%s" % n for n in nodes]
        ret = hmget_helper(self._rs, query, self.num_parts)
        v = []
        eid = []
        for buff in ret:
            if buff is not None:
                npret = np.frombuffer(
                    buff, dtype="int64").reshape([-1, 2]).astype("int64")
                v.append(npret[:, 0])
                eid.append(npret[:, 1])
            else:
                v.append(np.array([], dtype="int64"))
                eid.append(np.array([], dtype="int64"))
        if return_eids:
            return np.array(v), np.array(eid)
        else:
            return np.array(v)

    def get_edges_by_id(self, eids):
        """get_edges_by_id"""
        queries = ["e:%s" % e for e in eids]
        ret = hmget_helper(self._rs, queries, self.num_parts)
        o = np.asarray(ret, dtype="int64")
        dst = o % self.num_nodes
        src = o // self.num_nodes
        data = np.hstack(
            [src.reshape([-1, 1]), dst.reshape([-1, 1])]).astype("int64")
        return data

    def get_node_feat_by_id(self, key, nodes):
        """get_node_feat_by_id"""
        queries = ["nf:%s:%i" % (key, nid) for nid in nodes]
        ret = hmget_helper(self._rs, queries, self.num_parts)
        ret = b"".join(ret)
        data = np.frombuffer(ret, dtype=self.node_feat_dtype(key))
        data = data.reshape(self.node_feat_shape(key))
        return data

    def get_edge_feat_by_id(self, key, eids):
        """get_edge_feat_by_id"""
        queries = ["ef:%s:%i" % (key, e) for e in eids]
        ret = hmget_helper(self._rs, queries, self.num_parts)
        ret = b"".join(ret)
        data = np.frombuffer(ret, dtype=self.edge_feat_dtype(key))
        data = data.reshape(self.edge_feat_shape(key))
        return data

    def subgraph(self, nodes, eid, edges=None):
        """Generate subgraph with nodes and edge ids.

        This function will generate a :code:`pgl.graph.Subgraph` object and
        copy all corresponding node and edge features. Nodes and edges will
        be reindex from 0.

        WARNING: ALL NODES IN EID MUST BE INCLUDED BY NODES

        Args:
            nodes: Node ids which will be included in the subgraph.

            eid: Edge ids which will be included in the subgraph.

        Return:
            A :code:`pgl.graph.Subgraph` object.
        """
        reindex = {}

        for ind, node in enumerate(nodes):
            reindex[node] = ind

        if edges is None:
            edges = self.get_edges_by_id(eid)
        else:
            edges = np.array(edges, dtype="int64")

        sub_edges = graph_kernel.map_edges(
            np.arange(
                len(edges), dtype="int64"), edges, reindex)

        sub_edge_feat = {}
        for key, _, _ in self.edge_feat_info():
            sub_edge_feat[key] = self.get_edge_feat_by_id(key, eid)

        sub_node_feat = {}
        for key, _, _ in self.node_feat_info():
            sub_node_feat[key] = self.get_node_feat_by_id(key, nodes)

        subgraph = pgraph.SubGraph(
            num_nodes=len(nodes),
            edges=sub_edges,
            node_feat=sub_node_feat,
            edge_feat=sub_edge_feat,
            reindex=reindex)
        return subgraph

    def node_batch_iter(self, batch_size, shuffle=True):
        """Node batch iterator

        Iterate all node by batch.

        Args:
            batch_size: The batch size of each batch of nodes.

            shuffle: Whether shuffle the nodes.

        Return:
            Batch iterator
        """
        perm = np.arange(self.num_nodes, dtype="int64")
        if shuffle:
            np.random.shuffle(perm)
        start = 0
        while start < self._num_nodes:
            yield perm[start:start + batch_size]
            start += batch_size
