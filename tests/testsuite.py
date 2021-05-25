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

import numpy as np
import pgl


def create_random_graph():
    dim = 8
    num_nodes = np.random.randint(low=8, high=16)
    edges = np.random.randint(
        low=0,
        high=num_nodes,
        size=[np.random.randint(
            low=num_nodes * 3, high=num_nodes * 4), 2])
    nfeat = np.random.randn(num_nodes, dim)
    efeat = np.random.randn(len(edges), dim)

    g = pgl.Graph(
        edges=edges,
        num_nodes=num_nodes,
        node_feat={'nfeat': nfeat},
        edge_feat={'efeat': efeat})
    return g

def create_random_bigraph():
    dim = 4
    src_num_nodes = np.random.randint(low=2, high=10)
    dst_num_nodes = np.random.randint(low=2, high=10)
    edges_size = np.random.randint(low=1, high=10)
    edges_src = np.random.randint(
        low=1,
        high=src_num_nodes,
        size=[edges_size, 1])

    edges_dst = np.random.randint(
        low=1,
        high=dst_num_nodes,
        size=[edges_size, 1])

    edges = np.hstack([edges_src, edges_dst])

    src_nfeat = np.random.randn(src_num_nodes, dim)
    dst_nfeat = np.random.randn(dst_num_nodes, dim)
    efeat = np.random.randn(len(edges), dim)

    g = pgl.BiGraph(
        edges=edges,
        src_num_nodes=src_num_nodes,
        dst_num_nodes=dst_num_nodes,
        src_node_feat={'nfeat': src_nfeat},
        dst_node_feat={'nfeat': dst_nfeat},
        edge_feat={'efeat': efeat})
    return g
