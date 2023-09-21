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
"""Implement Graph Partition Methods
"""

import math

import numpy as np
from pgl.graph_kernel import metis_partition as _metis_partition
from pgl.utils.helper import check_is_tensor
from pgl.utils.logger import log


def _metis_weight_scale(X):
    """Ensure X is postive integers.
    """
    X_min = np.min(X)
    X_max = np.max(X)
    X_scaled = (X - X_min) / (X_max - X_min + 1e-5)
    X_scaled = (X_scaled * 1000).astype("int64") + 1
    assert np.any(
        X_scaled > 0), "The weight of METIS input must be postive integers"
    return X_scaled


def metis_partition(graph, npart, node_weights=None, edge_weights=None):
    """Perform Metis Partition over graph.
    
    Graph Partition with third-party library METIS. Input graph, npart, node_weights and 
    edge_weights. Return a `numpy.ndarray` denotes which cluster the node belongs to.

    Args:

        graph (pgl.Graph): The input graph for partition.

        npart (int): The number of part in the final cluster.
  
        node_weights (optional): The node weights for each node. We will automatically use (MinMaxScaler + 1) * 1000
                                to convert the array into postive integers.

        edge_weights (optional): The edge weights for each node. We will automatically use (MinMaxScaler + 1) * 1000
                                to convert the array into postive integers.

    Returns:

        part_id (numpy.ndarray): An int64 numpy array with shape [num_nodes, ] denotes the cluster id.

    """

    log.warning("The input graph of metis_partition should be undirected.")

    if npart == 1:
        return np.zeros(graph.num_nodes, dtype=np.int64)

    csr = graph.adj_dst_index.numpy(inplace=False)
    indptr = csr._indptr
    v = csr._sorted_v
    sorted_eid = csr._sorted_eid
    if edge_weights is not None:
        if check_is_tensor(edge_weights):
            edge_weights = edge_weights.numpy()
        edge_weights = edge_weights[sorted_eid.tolist()]
        edge_weights = _metis_weight_scale(edge_weights)

    if node_weights is not None:
        if check_is_tensor(node_weights):
            node_weights = node_weights.numpy()
        node_weights = _metis_weight_scale(node_weights)

    # TODO: @Yelrose support recursive METIS
    # use K-way metis; recursive metis always core dump 
    part_id = _metis_partition(
        graph.num_nodes,
        indptr,
        v,
        nparts=npart,
        edge_weights=edge_weights,
        node_weights=node_weights,
        recursive=False)
    return part_id


def random_partition(graph, npart):
    """Perform Random Partition over graph.
  
    For random partition, we try to make each part the same size.
    Return a `numpy.ndarray` denotes which cluster the node belongs to.

    Args:

        graph (pgl.Graph): The input graph for partition

        npart (int): The number of part in the final cluster.

    Returns:

        part_id (numpy.ndarray): An int64 numpy array with shape [num_nodes, ] denotes the cluster id.         
    
    """

    if npart == 1:
        return np.zeros(graph.num_nodes, dtype=np.int64)

    cs = int(math.ceil(graph.num_nodes / npart))
    part_list = []
    for i in range(npart):
        part_list.extend([i] * cs)
    part_list = part_list[:graph.num_nodes]
    part_id = np.array(part_list, dtype=np.int64)
    np.random.shuffle(part_id)

    return part_id
