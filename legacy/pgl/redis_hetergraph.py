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
"""redis_hetergraph"""

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
from pgl import heter_graph
import pgl.redis_graph as rg


class RedisHeterGraph(rg.RedisGraph):
    """Redis Heterogeneous Graph"""

    def __init__(self, name, edge_types, redis_config, num_parts):
        super(RedisHeterGraph, self).__init__(name, redis_config, num_parts)
        self._num_edges = {}
        self.edge_types = edge_types
        self.e_type = None

        self._edge_feat_info = {}
        self._edge_feat_dtype = {}
        self._edge_feat_shape = {}

    def num_edges_by_type(self, e_type):
        """get edge number by specified edge type"""
        if e_type not in self._num_edges:
            self._num_edges[e_type] = int(
                self._rs.get("%s:num_edges" % e_type))

        return self._num_edges[e_type]

    def num_edges(self):
        """num_edges"""
        num_edges = {}
        for e_type in self.edge_types:
            num_edges[e_type] = self.num_edges_by_type(e_type)

        return num_edges

    def edge_feat_info_by_type(self, e_type):
        """get edge features information by specified edge type"""
        if e_type not in self._edge_feat_info:
            buff = self._rs.get("%s:ef:infos" % e_type)
            if buff is not None:
                self._edge_feat_info[e_type] = json.loads(buff.decode())
            else:
                self._edge_feat_info[e_type] = []
        return self._edge_feat_info[e_type]

    def edge_feat_info(self):
        """edge_feat_info"""
        edge_feat_info = {}
        for e_type in self.edge_types:
            edge_feat_info[e_type] = self.edge_feat_info_by_type(e_type)
        return edge_feat_info

    def edge_feat_shape(self, e_type, key):
        """edge_feat_shape"""
        if e_type not in self._edge_feat_shape:
            e_feat_shape = {}
            for k, shape, _ in self.edge_feat_info()[e_type]:
                e_feat_shape[k] = shape
            self._edge_feat_shape[e_type] = e_feat_shape
        return self._edge_feat_shape[e_type][key]

    def edge_feat_dtype(self, e_type, key):
        """edge_feat_dtype"""
        if e_type not in self._edge_feat_dtype:
            e_feat_dtype = {}
            for k, _, dtype in self.edge_feat_info()[e_type]:
                e_feat_dtype[k] = dtype
            self._edge_feat_dtype[e_type] = e_feat_dtype
        return self._edge_feat_dtype[e_type][key]

    def sample_predecessor(self, e_type, nodes, max_degree, return_eids=False):
        """sample predecessor with the specified edge type"""
        query = ["%s:d:%s" % (e_type, n) for n in nodes]
        rets = rg.hmget_sample_helper(self._rs, query, self.num_parts,
                                      max_degree)
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

    def sample_successor(self, e_type, nodes, max_degree, return_eids=False):
        """sample successor with the specified edge type"""
        query = ["%s:s:%s" % (e_type, n) for n in nodes]
        rets = rg.hmget_sample_helper(self._rs, query, self.num_parts,
                                      max_degree)
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

    def predecessor(self, e_type, nodes, return_eids=False):
        """predecessor with the specified edge type"""
        query = ["%s:d:%s" % (e_type, n) for n in nodes]
        ret = rg.hmget_helper(self._rs, query, self.num_parts)
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

    def successor(self, e_type, nodes, return_eids=False):
        """successor with the specified edge type"""
        query = ["%s:s:%s" % (e_type, n) for n in nodes]
        ret = rg.hmget_helper(self._rs, query, self.num_parts)
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

    def get_edges_by_id(self, e_type, eids):
        """get_edges_by_id"""
        queries = ["%s:e:%s" % (e_type, e) for e in eids]
        ret = rg.hmget_helper(self._rs, queries, self.num_parts)
        o = np.asarray(ret, dtype="int64")
        dst = o % self.num_nodes
        src = o // self.num_nodes
        data = np.hstack(
            [src.reshape([-1, 1]), dst.reshape([-1, 1])]).astype("int64")
        return data

    def get_edge_feat_by_id(self, e_type, key, eids):
        """get_edge_feat_by_id"""
        queries = ["%s:ef:%s:%i" % (e_type, key, e) for e in eids]
        ret = rg.hmget_helper(self._rs, queries, self.num_parts)
        if ret is None:
            return None
        else:
            ret = b"".join(ret)
            data = np.frombuffer(ret, dtype=self.edge_feat_dtype(e_type, key))
            data = data.reshape(self.edge_feat_shape(e_type, key))
            return data

    def get_node_types(self, nodes):
        """get_node_types """
        queries = ["nt:%i" % n for n in nodes]
        ret = rg.hmget_helper(self._rs, queries, self.num_parts)
        node_types = []
        for buff in ret:
            if buff:
                node_types.append(buff.decode())
            else:
                node_types = None
        return node_types

    def subgraph(self, nodes, eid, edges=None):
        """Generate heterogeneous subgraph with nodes and edge ids.

        WARNING: ALL NODES IN EID MUST BE INCLUDED BY NODES

        Args:
            nodes: Node ids which will be included in the subgraph.

            eid: Edge ids which will be included in the subgraph.

        Return:
            A :code:`pgl.heter_graph.Subgraph` object.
        """
        reindex = {}

        for ind, node in enumerate(nodes):
            reindex[node] = ind

        _node_types = self.get_node_types(nodes)
        if _node_types is None:
            node_types = None
        else:
            node_types = []
            for idx, t in zip(nodes, _node_types):
                node_types.append([reindex[idx], t])

        if edges is None:
            edges = {}
            for e_type, eid_list in eid.items():
                edges[e_type] = self.get_edges_by_id(e_type, eid_list)

        sub_edges = {}
        for e_type, edges_list in edges.items():
            sub_edges[e_type] = graph_kernel.map_edges(
                np.arange(
                    len(edges_list), dtype="int64"), edges_list, reindex)

        sub_edge_feat = {}
        for e_type, edge_feat_info in self.edge_feat_info().items():
            type_edge_feat = {}
            for key, _, _ in edge_feat_info:
                type_edge_feat[key] = self.get_edge_feat_by_id(e_type, key,
                                                               eid)
            sub_edge_feat[e_type] = type_edge_feat

        sub_node_feat = {}
        for key, _, _ in self.node_feat_info():
            sub_node_feat[key] = self.get_node_feat_by_id(key, nodes)

        subgraph = heter_graph.SubHeterGraph(
            num_nodes=len(nodes),
            edges=sub_edges,
            node_types=node_types,
            node_feat=sub_node_feat,
            edge_feat=sub_edge_feat,
            reindex=reindex)
        return subgraph
