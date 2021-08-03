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
import sys
sys.path.append("../")
import time
import warnings
import functools
import numpy as np
from collections import defaultdict, OrderedDict

import paddle.distributed.fleet as fleet
import pgl
from pgl.graph_kernel import skip_gram_gen_pair
from pgl.utils.logger import log
from pgl.distributed import DistGraphClient, DistGraphServer
from pgl.utils.data import Dataloader, StreamDataset
from pgl.distributed import helper

from utils.config import prepare_config
from utils.ego_sampling import graphsage_sampling
from datasets.node import NodeGenerator
from datasets.walk import WalkGenerator
from datasets.ego_graph import EgoGraphGenerator
from datasets.pair import PairGenerator

__all__ = [
    "Generator",
    "WalkBasedDataset",
    "CollateFn",
    "SageDataset",
    "SageCollateFn",
]


class Generator(object):
    def __init__(self):
        self.generator = None

    def apply(self, gen):
        ret_gen = functools.partial(gen, generator=self.generator)
        ret = type(self).from_generator_func(ret_gen)
        return ret

    @classmethod
    def from_generator_func(cls, _gen):
        ret = cls()
        ret.generator = _gen
        return ret

    def __call__(self):
        for data in self.generator():
            yield data


class WalkBasedDataset(StreamDataset):
    def __init__(self, config, ip_list_file, mode="train", **kwargs):
        self.config = config
        self.ip_list_file = ip_list_file
        self.mode = mode
        self.kwargs = kwargs

    def __iter__(self):
        mode = "_%s" % self.mode
        generator = getattr(self, mode)

        for data in generator():
            yield data

    def _train(self):
        log.info("gpu train data generator")
        client_id = os.getpid()
        graph = DistGraphClient(self.config, self.config.shard_num,
                                self.ip_list_file, client_id)
        rank = self._worker_info.fid
        nrank = self._worker_info.num_workers

        pipeline = [
            NodeGenerator(
                self.config, graph, rank=rank, nrank=nrank), WalkGenerator(
                    self.config, graph, rank=rank, nrank=nrank), PairGenerator(
                        self.config, graph, rank=rank, nrank=nrank)
        ]

        self.generator = Generator()
        for p in pipeline:
            self.generator = self.generator.apply(p)

        cc = 0
        for data in self.generator():
            yield data
            cc += 1

    def _infer(self):
        log.info("infer data generator")
        client_id = os.getpid()
        graph = DistGraphClient(self.config, self.config.shard_num,
                                self.ip_list_file, client_id)
        rank = self._worker_info.fid
        nrank = self._worker_info.num_workers

        generator = NodeGenerator(self.config, graph, rank=rank, nrank=nrank)

        for batch_nodes in generator():
            for nid in batch_nodes[0]:
                yield [nid, nid]

    def _distcpu_infer(self):
        log.info("distcpu infer data generator")
        client_id = os.getpid()
        graph = DistGraphClient(self.config, self.config.shard_num,
                                self.ip_list_file, client_id)

        # total number rank = total fleet workers * dataloader workers
        nrank = self._worker_info.num_workers * fleet.worker_num()
        rank = self._worker_info.fid + self._worker_info.num_workers * fleet.worker_index(
        )
        log.info("rank %s in total nrank %s" % (rank, nrank))

        generator = NodeGenerator(self.config, graph, rank=rank, nrank=nrank)

        for batch_nodes in generator():
            for nid in batch_nodes[0]:
                yield [nid, nid]

    def _distcpu_train(self):
        log.info("distcpu data generator")
        client_id = os.getpid()
        graph = DistGraphClient(self.config, self.config.shard_num,
                                self.ip_list_file, client_id)
        # total number rank = total fleet workers * dataloader workers
        nrank = self._worker_info.num_workers * fleet.worker_num()
        rank = self._worker_info.fid + self._worker_info.num_workers * fleet.worker_index(
        )
        log.info("rank %s in total nrank %s" % (rank, nrank))

        pipeline = [
            NodeGenerator(
                self.config, graph, rank=rank, nrank=nrank), WalkGenerator(
                    self.config, graph, rank=rank, nrank=nrank), PairGenerator(
                        self.config, graph, rank=rank, nrank=nrank)
        ]

        self.generator = Generator()
        for p in pipeline:
            self.generator = self.generator.apply(p)

        cc = 0
        for epoch in range(self.config.epochs):
            log.info("epoch [%s] in rank [%s]" % (epoch, rank))
            for data in self.generator():
                yield data
                cc += 1


class CollateFn(object):
    def __init__(self, config=None, mode="gpu"):
        self.config = config
        self.mode = mode

    def __call__(self, batch_data):
        feed_dict = OrderedDict()
        src_list = []
        pos_list = []
        for src, pos in batch_data:
            src_list.append(src)
            pos_list.append(pos)

        if self.mode == "gpu":
            src_list = np.array(src_list, dtype="int64").reshape(-1, )
            pos_list = np.array(pos_list, dtype="int64").reshape(-1, )
        elif self.mode == "distcpu":
            src_list = np.array(src_list, dtype="int64").reshape(-1, 1)
            pos_list = np.array(pos_list, dtype="int64").reshape(-1, 1)

        feed_dict['src'] = src_list
        feed_dict['pos'] = pos_list

        bs = len(src_list)
        neg_idx = np.random.randint(
            low=0, high=bs, size=[bs * self.config.neg_num], dtype="int64")
        feed_dict['neg_idx'] = neg_idx

        if self.mode == "gpu":
            return feed_dict
        elif self.mode == "distcpu":
            return tuple(list(feed_dict.values()))


class SageDataset(StreamDataset):
    def __init__(self, config, ip_list_file, mode="train"):
        self.config = config
        self.ip_list_file = ip_list_file
        self.mode = mode

    def __iter__(self):
        mode = "_%s" % self.mode
        generator = getattr(self, mode)

        for data in generator():
            yield data

    def _train(self):
        log.info("gpu train data generator")
        client_id = os.getpid()
        graph = DistGraphClient(self.config, self.config.shard_num,
                                self.ip_list_file, client_id)
        rank = self._worker_info.fid
        nrank = self._worker_info.num_workers

        pipeline = [
            NodeGenerator(
                self.config, graph, rank=rank, nrank=nrank), WalkGenerator(
                    self.config, graph, rank=rank, nrank=nrank),
            EgoGraphGenerator(
                self.config, graph, rank=rank, nrank=nrank), PairGenerator(
                    self.config, graph, rank=rank, nrank=nrank)
        ]

        self.generator = Generator()
        for p in pipeline:
            self.generator = self.generator.apply(p)

        cc = 0
        for data in self.generator():
            yield data
            cc += 1

    def _infer(self):
        log.info("infer data generator")
        client_id = os.getpid()
        graph = DistGraphClient(self.config, self.config.shard_num,
                                self.ip_list_file, client_id)
        rank = self._worker_info.fid
        nrank = self._worker_info.num_workers
        self.edge_types = graph.get_edge_types()
        self.ntype_list = graph.get_node_types()

        for ntype in self.ntype_list:
            for batch_nodes in graph.node_batch_iter(
                    batch_size=100, node_type=ntype, rank=rank, nrank=nrank):

                ego_graphs, _ = graphsage_sampling(
                    graph,
                    batch_nodes,
                    self.config.infer_sample_num_list,
                    edge_types=self.edge_types)
                for ego in ego_graphs:
                    yield [ego, ego]

    def _distcpu_infer(self):
        log.info("distcpu infer data generator")
        client_id = os.getpid()
        graph = DistGraphClient(self.config, self.config.shard_num,
                                self.ip_list_file, client_id)
        self.edge_types = graph.get_edge_types()
        self.ntype_list = graph.get_node_types()

        # total number rank = total fleet workers * dataloader workers
        nrank = self._worker_info.num_workers * fleet.worker_num()
        rank = self._worker_info.fid + self._worker_info.num_workers * fleet.worker_index(
        )
        log.info("rank %s in total nrank %s" % (rank, nrank))

        for ntype in self.ntype_list:
            for batch_nodes in graph.node_batch_iter(
                    batch_size=100, node_type=ntype, rank=rank, nrank=nrank):

                ego_graphs, _ = graphsage_sampling(
                    graph,
                    batch_nodes,
                    self.config.infer_sample_num_list,
                    edge_types=self.edge_types)
                for ego in ego_graphs:
                    yield [ego, ego]

    def _distcpu_train(self):
        log.info("distcpu data generator")
        client_id = os.getpid()
        graph = DistGraphClient(self.config, self.config.shard_num,
                                self.ip_list_file, client_id)
        # total number rank = total fleet workers * dataloader workers
        nrank = self._worker_info.num_workers * fleet.worker_num()
        rank = self._worker_info.fid + self._worker_info.num_workers * fleet.worker_index(
        )
        log.info("rank %s in total nrank %s" % (rank, nrank))

        pipeline = [
            NodeGenerator(
                self.config, graph, rank=rank, nrank=nrank), WalkGenerator(
                    self.config, graph, rank=rank, nrank=nrank),
            EgoGraphGenerator(
                self.config, graph, rank=rank, nrank=nrank), PairGenerator(
                    self.config, graph, rank=rank, nrank=nrank)
        ]

        self.generator = Generator()
        for p in pipeline:
            self.generator = self.generator.apply(p)

        cc = 0
        for epoch in range(self.config.epochs):
            log.info("epoch [%s] in rank [%s]" % (epoch, rank))
            for data in self.generator():
                yield data
                cc += 1


class SageCollateFn(object):
    def __init__(self, config, mode="gpu"):
        self.config = config
        self.mode = mode
        if self.config.symmetry:
            self.symmetry = self.config.symmetry
        else:
            self.symmetry = False
        self.etype2files = helper.parse_files(self.config.etype2files)
        self.edge_type_list = helper.get_all_edge_type(self.etype2files,
                                                       self.symmetry)

    def __call__(self, batch_data):
        # Pair of EgoGraphs
        feed_dict = OrderedDict()

        src_list = []
        pos_list = []

        center_id = []
        graphs = []

        total_num_nodes = 0
        for src, pos in batch_data:
            center_id.append(total_num_nodes)
            graphs.append(src)
            total_num_nodes += src.num_nodes

            center_id.append(total_num_nodes)
            graphs.append(pos)
            total_num_nodes += pos.num_nodes

        graphs = pgl.Graph.batch(graphs)

        feed_dict['num_nodes'] = np.array([graphs.num_nodes], dtype="int64")

        for etype_id, etype in enumerate(self.edge_type_list):
            edges = graphs.edges[graphs.edge_feat["edge_type"] == etype_id]
            feed_dict['num_edges_%s' % etype] = np.array(
                [len(edges)], dtype="int64")
            feed_dict['edges_%s' % etype] = edges

        # the total node index of the subgraph
        if self.mode == "gpu":
            feed_dict["batch_node_index"] = graphs.node_feat[
                "node_id"].reshape(-1, )
        elif self.mode == "distcpu":
            feed_dict["batch_node_index"] = graphs.node_feat[
                "node_id"].reshape(-1, 1)
        else:
            raise ValueError(
                "[%s] mode is not recognized, it should be [gpu] or [distcpu]")

        # the center node index of the subgraph
        feed_dict['center_node_index'] = np.array(center_id, dtype="int64")

        bs = len(batch_data)
        neg_idx = np.random.randint(
            low=0, high=bs, size=[bs * self.config.neg_num], dtype="int64")
        feed_dict['neg_idx'] = neg_idx

        if self.mode == "gpu":
            return feed_dict
        elif self.mode == "distcpu":
            return tuple(list(feed_dict.values()))
