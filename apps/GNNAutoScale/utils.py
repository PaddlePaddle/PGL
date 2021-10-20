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
    """Generate different masks for train/validation/test mode.
    """
    mask = np.zeros(num_nodes, dtype=np.int32)
    mask[index] = 1
    return mask


def permute(data, feature, permutation):
    """Permute data and feature according to the input `permutation`.
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
    data.train_mask = paddle.to_tensor(data.train_mask[permutation])
    data.val_mask = paddle.to_tensor(data.val_mask[permutation])
    data.test_mask = paddle.to_tensor(data.test_mask[permutation])

    if len(data.y.shape) == 1:
        data.label = paddle.to_tensor(np.expand_dims(data.y, -1)[permutation])
    else:
        data.label = paddle.to_tensor(data.y[permutation])

    feature = feature[permutation]
    feature = paddle.to_tensor(feature, place=paddle.CUDAPinnedPlace())
    return data, feature


def process_batch_data(batch_data, feature=None, norm=None):
    """Process batch data here.
    """

    g = batch_data.subgraph
    batch_size = batch_data.batch_size
    n_id = batch_data.n_id
    offset = batch_data.offset
    count = batch_data.count

    g.tensor()
    offset = paddle.to_tensor(offset, place=paddle.CPUPlace())
    count = paddle.to_tensor(count, place=paddle.CPUPlace())

    if feature is not None:
        feature = feature[n_id]
        feature = paddle.to_tensor(feature)

    if norm is not None:
        norm = norm[n_id]
        norm = paddle.to_tensor(norm)

    n_id = paddle.to_tensor(n_id)

    return g, batch_size, n_id, offset, count, feature, norm


def compute_acc(logits, y, mask):
    """Compute accuracy for train/validation/test masks.
    """

    if mask is not None:
        true_index = paddle.nonzero(mask)
        logits = paddle.gather(logits, true_index)
        y = paddle.gather(y, true_index)

    return paddle.metric.accuracy(logits, y)


def time_wrapper(func_name):
    """Time counter wrapper
    """

    def decorate(func):
        """decorate func
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """wrapper func
            """
            ts = time.time()
            result = func(*args, **kwargs)
            te = time.time()
            costs = te - ts
            if costs < 1e-4:
                cost_str = '%f sec' % costs
            elif costs > 3600:
                cost_str = '%.4f sec (%.4f hours)' % (costs, costs / 3600.)
            else:
                cost_str = '%.4f sec' % costs
            print('[%s] func takes %s' % (func_name, cost_str))
            return result

        return wrapper

    return decorate
