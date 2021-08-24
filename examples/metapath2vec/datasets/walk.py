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
import numpy as np
from collections import defaultdict

from pgl.utils.logger import log
from pgl.distributed import DistGraphClient, DistGraphServer

from utils.config import prepare_config
from datasets.helper import stream_shuffle_generator, AsynchronousGenerator
from datasets.node import NodeGenerator
import datasets.sampling as Sampler


class WalkGenerator(object):
    def __init__(self, config, graph, mode, **kwargs):
        self.config = config
        self.graph = graph
        self.mode = mode
        self.kwargs = kwargs

        self.meta_path = self.config.meta_path
        self.first_node_type = self.config.first_node_type

        first_ntypes = self.first_node_type.split(";")
        meta_paths = self.meta_path.split(";")
        for fnt, mp in zip(first_ntypes, meta_paths):
            f = mp.split("-")[0].split("2")[0]
            assert f == fnt, "the first ntype in metapath [%s] " \
                             "is not equal to first ntype [%s]" % (f, fnt)

        self.rank = kwargs.get("rank", 0)
        self.nrank = kwargs.get("nrank", 1)

        self.node_generator = NodeGenerator(
            self.config,
            self.graph,
            self.mode,
            rank=self.rank,
            nrank=self.nrank)

        self.walk_batch_size = 400
        self.walk_stream_shuffle_size = 100000

    def __call__(self):
        iterval = (400000 * 24 // self.config.walk_len)
        walk_generator = self.base_walk_generator
        walk_generator = AsynchronousGenerator(walk_generator, maxsize=10000)

        walk_cc = 0
        for walks in stream_shuffle_generator(walk_generator,
                                              self.walk_batch_size,
                                              self.walk_stream_shuffle_size):
            walk_cc += len(walks)
            if walk_cc % iterval == 0 and self.rank == 0:
                try:
                    log.info("the walk length is [%s] in rank [%s]" \
                            % (len(walks[0]), self.rank))
                except Exception as e:
                    log.info(e)
            yield walks
        log.info("total [%s] number walks in rank [%s]" % (walk_cc, self.rank))

    def base_walk_generator(self):
        meta_paths = self.meta_path.split(";")
        for nodes, idx in self.node_generator():
            #  start = time.time()
            if self.config.walk_times is not None:
                walks = Sampler.metapath_randomwalk_with_walktimes(
                    self.graph, nodes, meta_paths[idx], self.config.walk_len,
                    self.config.walk_times)

            else:
                walks = Sampler.metapath_randomwalk(
                    self.graph, nodes, meta_paths[idx], self.config.walk_len)

            yield walks
            #  log.info("walk generate time: %s" % (time.time() - start))
