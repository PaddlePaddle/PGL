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
"""Graph Dataset
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import os
import pgl
import sys
import json

import numpy as np

from dataset.base_dataset import BaseDataGenerator
from pgl.sample import alias_sample
from pgl.sample import pinsage_sample
from pgl.sample import graphsage_sample
from pgl.sample import edge_hash
from pgl.sample import extract_edges_from_nodes


class RandomPartition(BaseDataGenerator):
    def __init__(self, batch_func, graph, num_cluster, epoch, shuffle=True):

        self.graph = graph
        self.num_cluster = num_cluster
        self.batch_func = batch_func

        self.epoch = epoch
        self.cluster_id = [
            np.random.randint(
                low=0, high=num_cluster, size=graph.num_nodes)
            for _ in range(epoch)
        ]

        super(RandomPartition, self).__init__(
            buf_size=1000,
            num_workers=num_cluster,
            batch_size=1,
            shuffle=False)

        self.line_examples = []
        for i in range(epoch):
            for j in range(num_cluster):
                self.line_examples.append((i, j))

    def batch_fn(self, batch_info):
        batch_info = batch_info[0]
        batch_no = batch_info[1]
        epoch = batch_info[0]
        perm = np.arange(0, self.graph.num_nodes)
        batch_nodes = perm[self.cluster_id[epoch] == batch_no]
        eids = extract_edges_from_nodes(self.graph, batch_nodes)
        sub_g = self.graph.subgraph(
            nodes=batch_nodes,
            eid=eids,
            with_node_feat=True,
            with_edge_feat=False)

        for key, value in self.graph.edge_feat.items():
            sub_g.edge_feat[key] = self.graph.edge_feat[key][eids]
        return self.batch_func(sub_g)
