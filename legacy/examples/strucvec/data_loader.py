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
"""
data_loader.py
"""
from pgl import graph
import numpy as np


class EdgeDataset():
    """
    The data load just read the edge file, at the same time reindex the source and destination.
    """

    def __init__(self, undirected=True, data_dir=""):
        self._undirected = undirected
        self._data_dir = data_dir
        self._load_edge_data()

    def _load_edge_data(self):
        node_sets = set()
        edges = []
        with open(self._data_dir, "r") as f:
            node_dict = dict()
            for line in f:
                src, dist = [
                    int(data) for data in line.strip("\n\r").split(" ")
                ]
                if src not in node_dict:
                    node_dict[src] = len(node_dict) + 1
                src = node_dict[src]
                if dist not in node_dict:
                    node_dict[dist] = len(node_dict) + 1
                dist = node_dict[dist]
                node_sets.add(src)
                node_sets.add(dist)
                edges.append((src, dist))
                if self._undirected:
                    edges.append((dist, src))

        num_nodes = len(node_sets)
        self.graph = graph.Graph(num_nodes=num_nodes + 1, edges=edges)
        self.nodes = np.array(list(node_sets))
        self.node_dict = node_dict
