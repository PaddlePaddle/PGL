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
    Some utility functions used in GNNAutoScale.
"""

import time
import functools

import numpy as np
import paddle
import pgl
from pgl.utils.logger import log
from paddle.fluid import core


def check_device():
    """Check whether the current device meets the GNNAutoScale conditions.
       This function must be called before the main program runs.
    """
    if not core.is_compiled_with_cuda():
        return False

    current_device = paddle.device.get_device()
    if current_device.startswith("cpu"):
        return False

    return True


def gen_mask(num_nodes, index):
    """Generate different masks for train/validation/test dataset. For input `index`, 
       the corresponding mask position will be set 1, otherwise 0.

    Args:

        num_nodes(int): Number of nodes in graph.

        index(numpy.ndarray): The index for train, validation or test dataset.

    Returns:

        mask(numpy.ndarray): Return masks for train/validation/test dataset.

    """
    mask = np.zeros(num_nodes, dtype=np.int32)
    mask[index] = 1
    return mask


def permute(data, feature, permutation, feat_gpu):
    """Permute data and feature according to the input `permutation`.

    Args:

        data(pgl.dataset): The input PGL dataset, for example: pgl.dataset.RedditDataset.
        
        feature(numpy.ndarray): Node feature of PGL graph.
        
        permutation(numpy.ndarray): New node permutation after graph partition.

        feat_gpu(bool): Whether to move node feature to GPU here.

    Returns:

        data(pgl.dataset): Return data after being permuted.

        feature: Return feature after being permuted.

    """
    edges = data.graph.edges
    reindex = {}
    for ind, node in enumerate(permutation):
        reindex[node] = ind
    new_edges = pgl.graph_kernel.map_edges(
        np.arange(
            len(edges), dtype="int64"), edges, reindex)
    g = pgl.Graph(edges=new_edges, num_nodes=data.graph.num_nodes)

    data.graph = g
    data.train_mask = data.train_mask[permutation]
    data.val_mask = data.val_mask[permutation]
    data.test_mask = data.test_mask[permutation]

    if len(data.y.shape) == 1:
        data.label = np.expand_dims(data.y, -1)[permutation]
    else:
        data.label = data.y[permutation]

    feature = feature[permutation]
    if feat_gpu:
        feature = paddle.to_tensor(feature)
    return data, feature


def process_batch_data(batch_data, feature=None, norm=None, only_nid=False):
    """Process batch data, mainly for turning numpy.array as paddle.Tensor.

    Args:

        batch_data(SubgraphData): Batch data returned from dataloader.

        feature(numpy.ndarray|paddle.Tensor): The permuted node feature. 

        norm(numpy.ndarray): Mainly used for GCN norm.

        only_nid(bool): If only_nid is True, just return batch_data.n_id.

    Returns:

        g(pgl.Graph): Return pgl.graph of batch_data, be in Tensor format.

        batch_size(int): Return batch size of the input batch_data.

        n_id(paddle.Tensor): Return node ids of the input batch_data.

        offset(paddle.Tensor): The begin indexes of graph partition parts in batch_data, should be placed on CPU.

        count(paddle.Tensor): The length of graph partition parts in batch_data, should be placed on CPU.

        feat(paddle.Tensor): The new indexed feat according to n_id. 

    """
    if only_nid:
        return batch_data.n_id

    g = batch_data.subgraph
    batch_size = batch_data.batch_size
    n_id = batch_data.n_id
    offset = batch_data.offset
    count = batch_data.count

    g.tensor()
    offset = paddle.to_tensor(offset, place=paddle.CPUPlace())
    count = paddle.to_tensor(count, place=paddle.CPUPlace())

    if feature is not None:
        feat = feature[n_id]
        feat = paddle.to_tensor(feat)
    else:
        feat = None

    if norm is not None:
        norm = norm[n_id]
        norm = paddle.to_tensor(norm)

    n_id = paddle.to_tensor(n_id)

    return g, batch_size, n_id, offset, count, feat, norm


def compute_acc(logits, y, mask):
    """Compute accuracy for train/validation/test masks.

    Args:
    
        logits (paddle.Tensor): logits returned from gnn models.

        y (numpy.ndarray): Labels of data samples.

        mask (numpy.ndarray): Mask of data samples for different datasets.

    """
    logits = logits.numpy()
    if mask is not None:
        true_index = np.nonzero(mask)[0]
        logits = logits[true_index]
        y = y[true_index].reshape([-1])
        acc = float(np.equal(
            np.argmax(
                logits, axis=-1), y).sum()) / len(true_index)
    return acc
