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

from pgl.graph_kernel import skip_gram_gen_pair
from pgl.utils.logger import log
from pgl.distributed import DistGraphClient, DistGraphServer
from pgl.utils.data import Dataloader, StreamDataset

from utils.config import prepare_config
from datasets.node import NodeGenerator
from datasets.walk import WalkGenerator


class PairGenerator(object):
    def __init__(self, config, graph, mode, **kwargs):
        self.config = config
        self.graph = graph
        self.mode = mode
        self.kwargs = kwargs
        self.rank = kwargs.get("rank", 0)
        self.nrank = kwargs.get("nrank", 1)

        self.walk_generator = WalkGenerator(
            self.config,
            self.graph,
            mode=self.mode,
            rank=self.rank,
            nrank=self.nrank)

    def __call__(self):
        iterval = 20000000 * 24 // self.config.walk_len
        pair_count = 0
        for walks in self.walk_generator():
            try:
                for walk in walks:
                    index = np.arange(0, len(walk), dtype="int64")
                    batch_s, batch_p = skip_gram_gen_pair(index,
                                                          self.config.win_size)
                    for s, p in zip(batch_s, batch_p):
                        yield walk[s], walk[p]
                        pair_count += 1
                        if pair_count % iterval == 0 and self.rank == 0:
                            log.info("[%s] pairs have been loaded in rank [%s]" \
                                    % (pair_count, self.rank))

            except Exception as e:
                log.exception(e)

        log.info("total [%s] pairs in rank [%s]" % (pair_count, self.rank))
