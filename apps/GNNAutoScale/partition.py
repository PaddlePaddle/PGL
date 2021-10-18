# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
    Graph partition methods for GNNAutoScale.
"""

import math
import numpy as np

import pgl
from pgl.partition import metis_partition


def random_partition(graph, npart, shuffle=True):
    """Randomly partition graph into small clusters.

    Args:

        graph(pgl.Graph): The input graph for partition.

        npart: The number of parts in the final graph partition.

        shuffle: Whether to shuffle the original node sequence.

    Returns:

        permutation: 

        split: 

    """

    num_nodes = graph.num_nodes

    if npart <= 1:
        permutation, ptr = np.arange(num_nodes), np.array([0, num_nodes])
    else:
        permutation = np.arange(0, num_nodes)
        if shuffle:
            np.random.shuffle(permutation)
        cs = int(math.ceil(num_nodes / npart))
        split = [
            cs * i if cs * i <= num_nodes else num_nodes
            for i in range(npart + 1)
        ]
        split = np.array(split)

    return permutation, split


def metis_graph_partition(graph, npart):
    """Using metis partition over graph.

    Args:

        graph(pgl.Graph): The input graph for partition.

        npart: The number of parts in the final graph partition.

    Returns:

        permutation: 

        split: 

    """

    part = metis_partition(graph, npart)
    permutation = np.argsort(part)

    split = np.zeros(npart + 1, dtype=np.int64)
    for i in range(npart):
        split[i + 1] = split[i] + len(np.where(part == i)[0])

    return permutation, split
