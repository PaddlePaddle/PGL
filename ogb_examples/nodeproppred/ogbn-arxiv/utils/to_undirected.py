#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""Arguments for configuration."""
from __future__ import absolute_import
from __future__ import unicode_literals

import paddle.fluid as fluid
import pgl
import numpy as np


def to_undirected(graph):
    inv_edges = np.zeros(graph.edges.shape)
    inv_edges[:, 0] = graph.edges[:, 1]
    inv_edges[:, 1] = graph.edges[:, 0]
    edges = np.vstack((graph.edges, inv_edges))
    g = pgl.graph.Graph(num_nodes=graph.num_nodes, edges=edges)
    for k, v in graph._edge_feat.items():
        g._edge_feat[k] = np.vstack((v, v))
    for k, v in graph._node_feat.items():
        g._node_feat[k] = v
    return g
