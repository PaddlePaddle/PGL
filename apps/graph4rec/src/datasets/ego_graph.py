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
from utils.ego_sampling import graphsage_sampling


class EgoGraphGenerator(object):
    def __init__(self, config, graph, **kwargs):
        self.config = config
        self.graph = graph
        self.kwargs = kwargs
        self.edge_types = self.graph.get_edge_types()

    def __call__(self, generator):
        """Input Batch of Walks
        """
        for walks in generator():
            # unique walk
            nodes = []
            for walk in walks:
                nodes.extend(walk)

            ego_graphs, _ = graphsage_sampling(
                self.graph,
                nodes,
                self.config.sample_num_list,
                edge_types=self.edge_types)

            start = 0
            egos = []
            for walk in walks:
                egos.append(ego_graphs[start:start + len(walk)])
                start += len(walk)
            yield egos
