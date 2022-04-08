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
from pgl.utils.logger import log

from utils.config import prepare_config
from datasets.node import NodeGenerator
from datasets.walk import WalkGenerator
from datasets.ego_graph import ego_graph_sample, EgoGraphGenerator
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

        pipeline = []
        pipeline.append(
            NodeGenerator(
                self.config, graph, rank=rank, nrank=nrank))
        pipeline.append(
            WalkGenerator(
                self.config, graph, rank=rank, nrank=nrank))
        pipeline.append(
            EgoGraphGenerator(
                self.config, graph, rank=rank, nrank=nrank))
        pipeline.append(
            PairGenerator(
                self.config, graph, rank=rank, nrank=nrank))

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

        pipeline = []
        pipeline.append(
            NodeGenerator(
                self.config,
                graph,
                rank=rank,
                nrank=nrank,
                gen_mode="infer_node_generator"))
        pipeline.append(
            WalkGenerator(
                self.config,
                graph,
                rank=rank,
                nrank=nrank,
                gen_mode="infer_walk_generator"))
        pipeline.append(
            EgoGraphGenerator(
                self.config, graph, rank=rank, nrank=nrank))

        generator = Generator()
        for p in pipeline:
            generator = generator.apply(p)

        cc = 0
        for walks in generator():
            for egos in walks:
                for ego in egos:
                    yield [ego, ego]
                    cc += 1

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

        pipeline = []
        pipeline.append(
            NodeGenerator(
                self.config,
                graph,
                rank=rank,
                nrank=nrank,
                gen_mode="infer_node_generator"))
        pipeline.append(
            WalkGenerator(
                self.config,
                graph,
                rank=rank,
                nrank=nrank,
                gen_mode="infer_walk_generator"))
        pipeline.append(
            EgoGraphGenerator(
                self.config, graph, rank=rank, nrank=nrank))

        generator = Generator()
        for p in pipeline:
            generator = generator.apply(p)

        cc = 0
        for walks in generator():
            for egos in walks:
                for ego in egos:
                    yield [ego, ego]
                    cc += 1

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

        pipeline = []
        pipeline.append(
            NodeGenerator(
                self.config, graph, rank=rank, nrank=nrank))
        pipeline.append(
            WalkGenerator(
                self.config, graph, rank=rank, nrank=nrank))
        pipeline.append(
            EgoGraphGenerator(
                self.config, graph, rank=rank, nrank=nrank))
        pipeline.append(
            PairGenerator(
                self.config, graph, rank=rank, nrank=nrank))

        self.generator = Generator()
        for p in pipeline:
            self.generator = self.generator.apply(p)

        cc = 0
        for epoch in range(self.config.epochs):
            log.info("epoch [%s] in rank [%s]" % (epoch, rank))
            for data in self.generator():
                yield data
                cc += 1


def index2segment_id(num_count):
    """
    num_count: [1, 2, 1]
    return:
        [0, 1, 1, 2]
    """
    index = np.cumsum(num_count, dtype="int64")
    index = np.insert(index, 0, 0)

    segments = np.zeros(index[-1] + 1, dtype="int64")
    index = index[:-1]
    segments[index] += 1
    segments = np.cumsum(segments)[:-1] - 1
    return segments


class CollateFn(object):
    def __init__(self, config=None, mode="gpu"):
        self.config = config
        self.mode = mode

    def __call__(self, batch_data):
        feed_dict = OrderedDict()
        for slot in self.config.slots:
            feed_dict[slot] = []
            feed_dict["%s_info" % slot] = []

        node_id_list = []
        offset = 0
        for src, pos in batch_data:
            node_id_list.extend(src.node_id)
            node_id_list.extend(pos.node_id)

            for slot in self.config.slots:
                feed_dict[slot].extend(src.feature[slot][0])
                segments = np.array(
                    src.feature[slot][1], dtype="int64") + offset
                feed_dict["%s_info" % slot].append(segments)

                feed_dict[slot].extend(pos.feature[slot][0])
                segments = np.array(
                    pos.feature[slot][1], dtype="int64") + offset + 1
                feed_dict["%s_info" % slot].append(segments)

            offset += 2

        for slot in self.config.slots:
            if self.mode == "gpu":
                feed_dict[slot] = np.array(
                    feed_dict[slot], dtype="int64").reshape(-1, )
            elif self.mode == "distcpu":
                feed_dict[slot] = np.array(
                    feed_dict[slot], dtype="int64").reshape(-1, 1)

            feed_dict["%s_info" % slot] = np.concatenate(feed_dict[
                "%s_info" % slot]).reshape(-1, )

        if self.mode == "gpu":
            node_id_list = np.array(node_id_list, dtype="int64").reshape(-1, )
        elif self.mode == "distcpu":
            node_id_list = np.array(node_id_list, dtype="int64").reshape(-1, 1)

        feed_dict['node_id'] = node_id_list

        bs = len(batch_data)
        neg_idx = np.random.randint(
            low=0, high=bs * 2, size=[bs * self.config.neg_num], dtype="int64")
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

        pipeline = []
        pipeline.append(
            NodeGenerator(
                self.config, graph, rank=rank, nrank=nrank))
        pipeline.append(
            WalkGenerator(
                self.config, graph, rank=rank, nrank=nrank))
        pipeline.append(
            EgoGraphGenerator(
                self.config, graph, rank=rank, nrank=nrank))
        pipeline.append(
            PairGenerator(
                self.config, graph, rank=rank, nrank=nrank))

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

        pipeline = []
        pipeline.append(
            NodeGenerator(
                self.config,
                graph,
                rank=rank,
                nrank=nrank,
                gen_mode="infer_node_generator"))
        pipeline.append(
            WalkGenerator(
                self.config,
                graph,
                rank=rank,
                nrank=nrank,
                gen_mode="infer_walk_generator"))
        pipeline.append(
            EgoGraphGenerator(
                self.config,
                graph,
                rank=rank,
                nrank=nrank,
                sample_list=self.config.infer_sample_num_list))

        generator = Generator()
        for p in pipeline:
            generator = generator.apply(p)

        cc = 0
        for walks in generator():
            for egos in walks:
                for ego in egos:
                    yield [ego, ego]
                    cc += 1

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

        pipeline = []
        pipeline.append(
            NodeGenerator(
                self.config,
                graph,
                rank=rank,
                nrank=nrank,
                gen_mode="infer_node_generator"))
        pipeline.append(
            WalkGenerator(
                self.config,
                graph,
                rank=rank,
                nrank=nrank,
                gen_mode="infer_walk_generator"))
        pipeline.append(
            EgoGraphGenerator(
                self.config,
                graph,
                rank=rank,
                nrank=nrank,
                sample_list=self.config.infer_sample_num_list))

        generator = Generator()
        for p in pipeline:
            generator = generator.apply(p)

        cc = 0
        for walks in generator():
            for egos in walks:
                for ego in egos:
                    yield [ego, ego]
                    cc += 1

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

        pipeline = []
        pipeline.append(
            NodeGenerator(
                self.config, graph, rank=rank, nrank=nrank))
        pipeline.append(
            WalkGenerator(
                self.config, graph, rank=rank, nrank=nrank))
        pipeline.append(
            EgoGraphGenerator(
                self.config, graph, rank=rank, nrank=nrank))
        pipeline.append(
            PairGenerator(
                self.config, graph, rank=rank, nrank=nrank))

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
        for slot in self.config.slots:
            feed_dict[slot] = []
            feed_dict["%s_info" % slot] = []

        center_id = []
        graphs = []

        offset = 0
        total_num_nodes = 0
        for src, pos in batch_data:
            center_id.append(total_num_nodes)
            graphs.append(src.graph)
            total_num_nodes += src.graph.num_nodes

            center_id.append(total_num_nodes)
            graphs.append(pos.graph)
            total_num_nodes += pos.graph.num_nodes

            for slot in self.config.slots:
                feed_dict[slot].extend(src.feature[slot][0])
                segments = np.array(
                    src.feature[slot][1], dtype="int64") + offset
                feed_dict["%s_info" % slot].append(segments)

                feed_dict[slot].extend(pos.feature[slot][0])
                segments = np.array(pos.feature[slot][1], dtype="int64")
                segments = segments + offset + len(
                    src.node_id)  # for pos offset 
                feed_dict["%s_info" % slot].append(segments)

            # add src and pos ego graphs' length after all slot finished adding
            offset = offset + len(src.node_id) + len(pos.node_id)

        for slot in self.config.slots:
            if self.mode == "gpu":
                feed_dict[slot] = np.array(
                    feed_dict[slot], dtype="int64").reshape(-1, )
            elif self.mode == "distcpu":
                feed_dict[slot] = np.array(
                    feed_dict[slot], dtype="int64").reshape(-1, 1)

            feed_dict["%s_info" % slot] = np.concatenate(feed_dict[
                "%s_info" % slot]).reshape(-1, )

        graphs = pgl.Graph.batch(graphs)

        feed_dict['num_nodes'] = np.array([graphs.num_nodes], dtype="int64")

        for etype_id, etype in enumerate(self.edge_type_list):
            edges = graphs.edges[graphs.edge_feat["edge_type"] == etype_id]
            feed_dict['num_edges_%s' % etype] = np.array(
                [len(edges)], dtype="int64")
            feed_dict['edges_%s' % etype] = edges

        # the total node index of the subgraph
        if self.mode == "gpu":
            feed_dict["origin_node_id"] = graphs.node_feat["node_id"].reshape(
                -1, )
        elif self.mode == "distcpu":
            feed_dict["origin_node_id"] = graphs.node_feat["node_id"].reshape(
                -1, 1)
        else:
            raise ValueError(
                "[%s] mode is not recognized, it should be [gpu] or [distcpu]")

        # the center node index of the subgraph
        feed_dict['center_node_id'] = np.array(center_id, dtype="int64")

        bs = len(batch_data)
        neg_idx = np.random.randint(
            low=0, high=bs * 2, size=[bs * self.config.neg_num], dtype="int64")
        feed_dict['neg_idx'] = neg_idx

        if self.mode == "gpu":
            return feed_dict
        elif self.mode == "distcpu":
            return tuple(list(feed_dict.values()))
