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

import os
import re
import sys
sys.path.append("../")
import time
import warnings
import numpy as np
from collections import defaultdict

import pgl
from pgl.utils.logger import log
from pgl.distributed import DistGraphClient, DistGraphServer

from utils.config import prepare_config
from datasets.helper import stream_shuffle_generator, AsynchronousGenerator
from datasets.node import NodeGenerator
import datasets.sampling as Sampler

NODE_FEAT_PATTERN = re.compile(r"[:,]")


class EgoInfo(object):
    def __init__(self,
                 node_id=None,
                 feature=[],
                 edges=[],
                 edges_type=[],
                 edges_weight=[],
                 graph=None):
        self.node_id = node_id
        self.feature = feature
        self.edges = edges
        self.edges_type = edges_type
        self.edges_weight = edges_weight
        self.graph = graph


def ego_graph_sample(graph, node_ids, samples, edge_types):
    """  Heter-EgoGraph                
                              ____ n1
                             /----n2
                     etype1 /---- n3
                   --------/-----n4
                  /        \-----n5
                 /
                /   
        start_n               
                \             /----n6
                 \  etype2 /---- n7
                   --------/-----n8
                          \-----n9

        TODO @Yelrose: Speed up and standarize to pgl.distributed.sampling
    """

    # All Nodes
    all_new_nodes = [node_ids]

    # Node Index
    all_new_nodes_index = [np.zeros_like(node_ids, dtype="int64")]

    # The Ego Index for each graph
    all_new_nodes_ego_index = [np.arange(0, len(node_ids), dtype="int64")]

    unique_nodes = set(node_ids)

    ego_graph_list = [
        EgoInfo(
            node_id=[n], edges=[], edges_type=[], edges_weight=[])
        for n in node_ids
    ]

    for sample in samples:
        cur_node_ids = all_new_nodes[-1]
        cur_node_ego_index = all_new_nodes_ego_index[-1]
        cur_node_index = all_new_nodes_index[-1]

        nxt_node_ids = []
        nxt_node_ego_index = []
        nxt_node_index = []
        for edge_type_id, edge_type in enumerate(edge_types):
            cur_succs = graph.sample_successor(
                cur_node_ids, max_degree=sample, edge_type=edge_type)
            for succs, ego_index, parent_index in zip(
                    cur_succs, cur_node_ego_index, cur_node_index):
                if len(succs) == 0:
                    succs = [0]
                ego = ego_graph_list[ego_index]

                for succ in succs:
                    unique_nodes.add(succ)
                    succ_index = len(ego.node_id)
                    ego.node_id.append(succ)
                    nxt_node_ids.append(succ)
                    nxt_node_ego_index.append(ego_index)
                    nxt_node_index.append(succ_index)
                    if succ == 0:
                        ego.edges.append((succ_index, succ_index))
                        ego.edges_type.append(edge_type_id)
                        ego.edges_weight.append(1.0 / len(succs))
                    else:
                        ego.edges.append((succ_index, parent_index))
                        ego.edges_type.append(edge_type_id)
                        ego.edges_weight.append(1.0 / len(succs))
        all_new_nodes.append(nxt_node_ids)
        all_new_nodes_index.append(nxt_node_index)
        all_new_nodes_ego_index.append(nxt_node_ego_index)

    for ego in ego_graph_list:
        pg = pgl.Graph(
            num_nodes=len(ego.node_id),
            edges=ego.edges,
            edge_feat={
                "edge_type": np.array(
                    ego.edges_type, dtype="int64"),
                "edge_weight": np.array(
                    ego.edges_weight, dtype="float32"),
            },
            node_feat={"node_id": np.array(
                ego.node_id, dtype="int64"), })
        ego.graph = pg
    return ego_graph_list, list(unique_nodes)


def get_slots_feat(graph, nodes, slots):
    nfeat_list = []
    for ntype in graph.get_node_types():
        nfeat_list.append(graph.get_node_feat(nodes, ntype, "s"))

    res = []
    for all_type_feat in zip(*nfeat_list):
        f = ""
        for feat in all_type_feat:
            if len(feat) > 0:
                f = feat
                break
        res.append(f)

    nfeat_dict = defaultdict(lambda: defaultdict(list))
    for nid, feat in zip(nodes, res):
        slot_feat = re.split(NODE_FEAT_PATTERN, feat)
        for k, v in zip(slot_feat[0::2], slot_feat[1::2]):
            try:
                v = int(v)
                nfeat_dict[nid][k].append(v)
            except Exception as e:
                continue

    return nfeat_dict


def make_slot_feat(node_id, slots, node_feat_dict):
    slot_dict = {}
    slot_cout = {}
    for slot in slots:
        slot_dict[slot] = ([], [])
        slot_cout[slot] = 0

    for n in node_id:
        nf = node_feat_dict[n]
        for slot in slots:
            if slot in nf:
                slot_dict[slot][0].extend(nf[slot])
                seg = np.zeros(len(nf[slot]), dtype="int64") + slot_cout[slot]
                slot_dict[slot][1].extend(seg)
            else:
                slot_dict[slot][0].append(0)
                slot_dict[slot][1].extend([slot_cout[slot]])
            slot_cout[slot] += 1

    return slot_dict


class EgoGraphGenerator(object):
    def __init__(self, config, graph, **kwargs):
        self.config = config
        self.graph = graph
        self.rank = kwargs.get("rank", 0)
        self.nrank = kwargs.get("nrank", 1)
        self.kwargs = kwargs
        self.edge_types = self.graph.get_edge_types()
        self.sample_num_list = kwargs.get("sample_list",
                                          self.config.sample_num_list)
        log.info("sample_num_list is %s" % repr(self.sample_num_list))

    def __call__(self, generator):
        self.generator = generator

        ego_generator = self.base_ego_generator
        ego_generator = AsynchronousGenerator(ego_generator, maxsize=1000)

        for data in ego_generator():
            yield data

    def base_ego_generator(self):
        """Input Batch of Walks
        """
        for walks in self.generator():
            # unique walk
            nodes = []
            for walk in walks:
                nodes.extend(walk)

            if self.config.sage_mode:  # GNN-based
                ego_graphs, uniq_nodes = ego_graph_sample(
                    self.graph,
                    nodes,
                    self.sample_num_list,
                    edge_types=self.edge_types)
            else:  # walk-based
                ego_graphs = [EgoInfo(node_id=[nid]) for nid in nodes]
                uniq_nodes = nodes

            if len(self.config.slots) > 0:
                nfeat_dict = get_slots_feat(self.graph, uniq_nodes,
                                            self.config.slots)
                for ego in ego_graphs:
                    ego.feature = make_slot_feat(ego.node_id,
                                                 self.config.slots, nfeat_dict)

            start = 0
            egos = []
            for walk in walks:
                egos.append(ego_graphs[start:start + len(walk)])
                start += len(walk)
            yield egos
