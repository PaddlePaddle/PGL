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


def random_partition(graph, npart, shuffle=True):
    """Randomly partition graph into small clusters.

    Args:

        graph(pgl.Graph): The input graph for partition.

        npart: The number of parts in the final graph partition.

        shuffle: Whether to shuffle the original node sequence.

    Returns:

        permutation (1D numpy array): New permutation of nodes in partition graph, and the shape is [num_nodes].

        part (1D numpy array): Help distinguish different parts of partition graphs, and the shape is [npart + 1].

    Example:
        - Suppose we have a graph, and its nodes are [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].
        - After random partition, we partition the graph into 4 parts, and the new permutation
          is [4, 6, 1, 5, 7, 0, 3, 2, 8, 9].
        - The part is [2, 3, 2, 3], which can distinguish different parts of partition graphs. 
          For example, part[0] is 2, which means nodes `4,6` construct a partition graph; part[1] is 3,
          which means nodes `1,5,7` construct a partition graph.

    """

    num_nodes = graph.num_nodes

    if npart <= 1:
        permutation, part = np.arange(num_nodes), np.array([0, num_nodes])
    else:
        permutation = np.arange(0, num_nodes)
        if shuffle:
            np.random.shuffle(permutation)
        cs = int(math.ceil(num_nodes / npart))
        part = [
            cs * i if cs * i <= num_nodes else num_nodes
            for i in range(npart + 1)
        ]
        part = np.array(part)

    return permutation, part
