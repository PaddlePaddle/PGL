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

import sys
import math

import numpy as np
import pgl
from pgl.utils.logger import log
from pgl.partition import random_partition, metis_partition


def random_graph_partition(graph, npart):
    """Randomly partition graph into small clusters.

    Args:

        graph (pgl.Graph): The input graph for partition.

        npart (int): The number of parts in the final graph partition.

    Returns:

        permutation (numpy.ndarray): An 1-D numpy array, which is the new permutation of nodes in partition graph, 
                                     and the shape is [num_nodes].

        part (numpy.ndarray): An 1-D numpy array, which helps distinguish different parts of partition graphs, 
                              and the shape is [npart + 1].

    Example:
        - Suppose we have a graph, and its nodes are [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].
        - After random partition, we partition the graph into 4 parts. Then we have new node `permutation`
          as [4, 6, 1, 5, 7, 0, 3, 2, 8, 9].
        - And we have `part` as [0, 3, 6, 9, 10], which can help distinguish different parts of partition graphs.
          For example, with (part[1]-part[0]) = 3, that means the number of nodes of the first partition graph is 3,
          and the corresponding nodes are [4, 6, 1]; with (part[4]-part[3]) = 1, that means the number of nodes of the 
          last partition graph is 1, and the corresponding node is [9].

    """

    num_nodes = graph.num_nodes

    if npart <= 1:
        permutation, part = np.arange(num_nodes), np.array([0, num_nodes])
    else:
        random_part = random_partition(graph, npart)
        permutation = np.argsort(random_part)

        part = np.zeros(npart + 1, dtype=np.int64)
        for i in range(npart):
            part[i + 1] = part[i] + len(np.where(random_part == i)[0])

    return permutation, part


def metis_graph_partition(graph, npart):
    """Using metis partition over graph into small clusters.

    Args:

        graph (pgl.Graph): The input graph for partition.

        npart (int): The number of parts in the final graph partition.

    Returns:

        permutation (numpy.ndarray): An 1-D numpy array, which is the new permutation of nodes in partition graph, 
                                     and the shape is [num_nodes]. 

        part (numpy.ndarray): An 1-D numpy array, which helps distinguish different parts of permutation, 
                              and the shape is [npart + 1].
 
    """

    if sys.platform == 'win32':
        raise NotImplementedError(
            f"We currently do not support metis partition on Windows system. "
            f"You can use random graph partition instead.")

    metis_part = metis_partition(graph, npart)
    permutation = np.argsort(metis_part)

    part = np.zeros(npart + 1, dtype=np.int64)
    for i in range(npart):
        part[i + 1] = part[i] + len(np.where(metis_part == i)[0])

    return permutation, part
