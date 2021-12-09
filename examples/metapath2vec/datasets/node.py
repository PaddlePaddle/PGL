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


class NodeGenerator(object):
    def __init__(self, config, graph, mode, **kwargs):
        self.config = config
        self.graph = graph
        self.mode = mode
        self.kwargs = kwargs

        self.batch_node_size = self.config.batch_node_size
        self.rank = kwargs.get("rank", 0)
        self.nrank = kwargs.get("nrank", 1)

    def __call__(self):
        generator = self.base_node_generator

        for nodes in generator():
            yield nodes

    def base_node_generator(self):
        ntype_list = self.graph.get_node_types()
        node_generators = {}
        for ntype in ntype_list:
            node_generators[ntype] = self.graph.node_batch_iter(
                batch_size=self.batch_node_size,
                shuffle=True,
                node_type=ntype,
                rank=self.rank,
                nrank=self.nrank)

        first_ntype_list = self.config.first_node_type.split(";")
        num_ntype = len(first_ntype_list)
        finished_node_types_set = set()

        cc = 0
        batch_count = defaultdict(lambda: 0)
        epoch_flag = False
        while True:
            idx = cc % num_ntype
            if idx not in finished_node_types_set:
                ntype = first_ntype_list[idx]
                try:
                    batch_nodes = next(node_generators[ntype])
                    batch_count[ntype] += 1
                    yield (batch_nodes, idx)
                except StopIteration as e:
                    log.info(e)
                    msg = "nodes of type [%s] finished with [%s] batch iteration in rank [%s]" \
                            % (ntype, batch_count[ntype], self.rank)
                    log.info(msg)
                    finished_node_types_set.add(idx)

                    msg = ""
                    for x in list(finished_node_types_set):
                        msg += " [%s]" % first_ntype_list[x]
                    log.info("%s node types have been finished in rank [%s]." \
                            % (msg, self.rank))

                    if len(finished_node_types_set) == num_ntype:
                        epoch_flag = True

                if epoch_flag:
                    break
            cc += 1
