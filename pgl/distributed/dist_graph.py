# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""
    This package implement the Distributed CPU Graph for 
    handling large scale graph data.
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np
from functools import partial

from paddle.framework import core
from pgl.utils.logger import log

from pgl.distributed import helper

__all__ = ['DistGraphServer', 'DistGraphClient']


def stream_shuffle_generator(dataloader,
                             server_idx,
                             batch_size,
                             shuffle_size=20000):
    """
    Args:
        dataloader: iterable dataset

        server_idx: int

        batch_size: int

        shuffle_size: int

    """
    buffer_list = []
    for nodes in dataloader(server_idx):
        if len(buffer_list) < shuffle_size:
            buffer_list.extend(nodes)
        else:
            random_ids = np.random.choice(
                len(buffer_list), size=len(nodes), replace=False)
            batch_nodes = [buffer_list[i] for i in random_ids]
            for ii, nid in enumerate(nodes):
                buffer_list[random_ids[ii]] = nid

            yield batch_nodes

    if len(buffer_list) > 0:
        np.random.shuffle(buffer_list)
        start = 0
        while True:
            batch_nodes = buffer_list[start:(start + batch_size)]
            start += batch_size
            if len(batch_nodes) > 0:
                yield batch_nodes
            else:
                break


class DistGraphServer(object):
    def __init__(self, config, shard_num, ip_config, server_id,
                 is_block=False):
        """
        Args:
            config: a yaml configure file or a dict of parameters.
            Below are some necessary hyper-parameters:
            ```
                etype2files: "u2e2t:./your_path/edges.txt"
                symmetry: True
                ntype2files: "u:./your_path/node_types.txt,t:./your_path/node_types.txt"

            ```

            shard_num: int, the sharding number of graph data

            ip_config: list of IP address or a path of IP configuration file
            
            For example, the following TXT shows a 4-machine configuration:

                172.31.50.123:8245
                172.31.50.124:8245
                172.31.50.125:8245
                172.31.50.126:8245

            server_id: int 

            is_block: bool, whether to block the server.

        """
        self.config = helper.load_config(config)
        self.shard_num = shard_num
        self.server_id = server_id
        self.is_block = is_block

        if self.config.symmetry:
            self.symmetry = self.config.symmetry
        else:
            self.symmetry = False

        self.ip_addr = helper.load_ip_addr(ip_config)

        self.ntype2files = helper.parse_files(self.config.ntype2files)
        self.node_type_list = list(self.ntype2files.keys())

        self.etype2files = helper.parse_files(self.config.etype2files)
        self.edge_type_list = helper.get_all_edge_type(self.etype2files,
                                                       self.symmetry)

        self._server = core.GraphPyServer()
        self._server.set_up(self.ip_addr, self.shard_num, self.node_type_list,
                            self.edge_type_list, self.server_id)

        if self.config.nfeat_info:
            for item in self.config.nfeat_info:
                self._server.add_table_feat_conf(*item)
        self._server.start_server(self.is_block)


class DistGraphClient(object):
    def __init__(self,
                 config,
                 shard_num,
                 ip_config,
                 client_id,
                 use_cache=False):
        """
        Args:
            config: a yaml configure file or a dict of parameters
            Below are some necessary hyper-parameters:
            ```
                etype2files: "u2e2t:./your_path/edges.txt"
                symmetry: True
                ntype2files: "u:./your_path/node_types.txt,t:./your_path/node_types.txt"

            ```

            shard_num: int, the sharding number of graph data

            ip_config: list of IP address or a path of IP configuration file
            
            For example, the following TXT shows a 4-machine configuration:

                172.31.50.123:8245
                172.31.50.124:8245
                172.31.50.125:8245
                172.31.50.126:8245

            client_id: int 

            use_cache: bool

        """
        self.config = helper.load_config(config)
        self.shard_num = shard_num
        self.client_id = client_id

        if self.config.symmetry:
            self.symmetry = self.config.symmetry
        else:
            self.symmetry = False

        if self.config.node_batch_stream_shuffle_size:
            self.stream_shuffle_size = self.config.node_batch_stream_shuffle_size
        else:
            warnings.warn("node_batch_stream_shuffle_size is not specified, "
                          "default value is 20000")
            self.stream_shuffle_size = 20000

        self.ip_addr = helper.load_ip_addr(ip_config)
        self.server_num = len(self.ip_addr.split(";"))

        if self.config.nfeat_info is not None:
            self.nfeat_info = helper.convert_nfeat_info(self.config.nfeat_info)
        else:
            self.nfeat_info = None

        self.ntype2files = helper.parse_files(self.config.ntype2files)
        self.node_type_list = list(self.ntype2files.keys())

        self.etype2files = helper.parse_files(self.config.etype2files)
        self.edge_type_list = helper.get_all_edge_type(self.etype2files,
                                                       self.symmetry)

        self._client = core.GraphPyClient()
        self._client.set_up(self.ip_addr, self.shard_num, self.node_type_list,
                            self.edge_type_list, self.client_id)
        self._client.start_client()

        if use_cache:
            for etype in self.edge_type_list:
                self._client.use_neighbors_sample_cache(etype, 100000, 6)

    def load_edges(self):
        for etype, file_or_dir in self.etype2files.items():
            file_list = [f for f in helper.get_files(file_or_dir)]
            filepath = ";".join(file_list)
            log.info("load edges of type %s from %s" % (etype, filepath))
            self._client.load_edge_file(etype, filepath, False)
            if self.symmetry:
                r_etype = helper.get_inverse_etype(etype)
                self._client.load_edge_file(r_etype, filepath, True)

    def load_node_types(self):
        for ntype, file_or_dir in self.ntype2files.items():
            file_list = [f for f in helper.get_files(file_or_dir)]
            filepath = ";".join(file_list)
            log.info("load nodes of type %s from %s" % (ntype, filepath))
            self._client.load_node_file(ntype, filepath)

    def sample_predecessor(self,
                           nodes,
                           max_degree,
                           edge_type=None,
                           return_weight=False,
                           return_edges=False,
                           split=True):
        """
        Args:
            nodes: list of node ID

            max_degree: int, sample number of each node

            edge_type: str, edge type

            return_weight: bool, if True, then the edge weight will return

            return_edges: bool, if True, the complete edges will return

            split: bool, if True, the neighbors of nodes will be splited to corresponding node
        """

        return self.sample_successor(
            nodes,
            max_degree,
            edge_type=edge_type,
            return_weight=return_weight,
            return_edges=return_edges,
            split=split)

    def sample_successor(self,
                         nodes,
                         max_degree,
                         edge_type=None,
                         return_weight=False,
                         return_edges=False,
                         split=True):
        """
        Args:
            nodes: list of node ID

            max_degree: int, sample number of each node

            edge_type: str, edge type

            return_weight: bool, if True, then the edge weight will return

            return_edges: bool, if True, the complete edges will return

            split: bool, if True, the neighbors of nodes will be splited to corresponding node
        """

        if edge_type is None:
            if len(self.edge_type_list) > 1:
                msg = "There are %s (%s) edge types in the Graph, " \
                        % (len(self.edge_type_list), self.edge_type_list)
                msg += "The argument of edge_type should be specified, "
                msg += "but got [None]."
                raise ValueError(msg)
            else:
                edge_type = self.edge_type_list[0]

        def _split_by_index(x, index):
            splited = [tmp.tolist() for tmp in np.split(x, index)]
            return splited

        # res[0][0]: neigbors (nodes)
        # res[0][1]: numpy split index
        # res[0][2]: src nodes
        # res[1]: edge weights
        res = self._client.batch_sample_neighboors(
            edge_type, nodes, max_degree, return_weight, return_edges)

        if return_edges:
            if return_weight:
                return np.array([res[0][2], res[0][0]]).T, res[1]
            else:
                return np.array([res[0][2], res[0][0]]).T
        else:
            if return_weight:
                if split:
                    neighs = _split_by_index(res[0][0], res[0][1])
                    weights = _split_by_index(res[1], res[0][1])
                    return neighs, weights
                else:
                    return res[0][0], res[1]
            else:
                if split:
                    neighs = _split_by_index(res[0][0], res[0][1])
                    return neighs
                else:
                    return res[0][0]

    def random_sample_nodes(self, node_type=None, size=1):
        """
        Args:
            node_type: str, if None, then random select one type from node_type_list

            size: int
        """
        if node_type is None:
            node_type = np.random.choice(self.node_type_list)
        sampled_nodes = []
        server_list = list(range(self.server_num))
        np.random.shuffle(server_list)
        left_size = size
        for server_idx in server_list:
            nodes = self._client.random_sample_nodes(node_type, server_idx,
                                                     left_size)
            sampled_nodes.extend(nodes)
            if len(sampled_nodes) >= size:
                break
            else:
                left_size = size - len(sampled_nodes)

        return sampled_nodes

    def _node_batch_iter_from_server(self,
                                     server_idx,
                                     batch_size,
                                     node_type,
                                     rank=0,
                                     nrank=1):
        assert batch_size > 0, \
                "batch_size should be larger than 0, but got %s <= 0" % batch_size
        assert server_idx >= 0 and server_idx < self.server_num, \
                "server_idx should be in range 0 <= server_idx < server_num, but got %s" \
                % server_idx
        start = rank
        step = nrank
        while True:
            res = self._client.pull_graph_list(node_type, server_idx, start,
                                               batch_size, step)
            start += (nrank * batch_size)
            nodes = [x.get_id() for x in res]

            if len(nodes) > 0:
                yield nodes
            if len(nodes) != batch_size:
                break

    def node_batch_iter(self,
                        batch_size,
                        node_type,
                        shuffle=True,
                        rank=0,
                        nrank=1):
        """
        Args:
            batch_size: int

            node_type: string

            shuffle: bool

            rank: int

            nrank: int
        """

        node_iter = partial(
            self._node_batch_iter_from_server,
            batch_size=batch_size,
            node_type=node_type,
            rank=rank,
            nrank=nrank)

        server_idx_list = list(range(self.server_num))
        np.random.shuffle(server_idx_list)
        for server_idx in server_idx_list:
            if shuffle:
                for nodes in stream_shuffle_generator(
                        node_iter, server_idx, batch_size,
                        self.stream_shuffle_size):
                    yield nodes
            else:
                for nodes in node_iter(server_idx):
                    yield nodes

    def get_node_feat(self, nodes, node_type, feat_names):
        """
        Args:
            nodes: list of node ID

            node_type: str, node type

            feat_names: the node feature name or a list of node feature name
        """

        flag = False
        if isinstance(feat_names, str):
            feat_names = [feat_names]
            flag = True
        elif isinstance(feat_names, list):
            pass
        else:
            raise TypeError(
                "The argument of feat_names should a node feature name "
                "or a list of node feature name. "
                "But got %s" % (type(feat_names)))

        byte_nfeat_list = self._client.get_node_feat(node_type, nodes,
                                                     feat_names)

        # convert bytes to dtype
        nfeat_list = []
        for fn_idx, fn in enumerate(feat_names):
            dtype, _ = self.nfeat_info[node_type][fn]
            if dtype == "string":
                f_list = [
                    item.decode("utf-8") for item in byte_nfeat_list[fn_idx]
                ]
            else:
                f_list = [
                    np.frombuffer(item, dtype)
                    for item in byte_nfeat_list[fn_idx]
                ]
            nfeat_list.append(f_list)

        if flag:
            return nfeat_list[0]
        else:
            return nfeat_list

    def stop_server(self):
        self._client.stop_server()

    def get_node_types(self):
        return self.node_type_list

    def get_edge_types(self):
        return self.edge_type_list
