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
import paddle
from paddle.fluid.framework import Variable

from pgl.utils import op


def check_is_tensor(*data):
    """Check if the given datas have paddle.Tensor
    """
    for d in data:
        if isinstance(d, paddle.Tensor) or isinstance(d, Variable):
            return True
    return False


def generate_segment_id_from_index(index):
    if check_is_tensor(index):
        zeros = paddle.zeros(index[-1] + 1, dtype="int32")
        index = index[:-1]
        segments = paddle.scatter(
            zeros, index, paddle.ones_like(
                index, dtype="int32"))
        segments = paddle.cumsum(segments)[:-1] - 1
        return segments
    else:
        segments = np.zeros(index[-1] + 1, dtype="int32")
        index = index[:-1]
        segments[index] += 1
        segments = np.cumsum(segments)[:-1] - 1
        return segments


def maybe_num_nodes(edges):
    """Guess the number of nodes from edges
    
    Args:

        edges: numpy.ndarry of paddle.Tensor

    Return:

        An int or paddle.Tensor about the number of nodes.
    """
    if isinstance(edges, Variable):
        return paddle.max(edges) + 1

    if len(edges) == 0:
        return 0

    if check_is_tensor(edges):
        return paddle.max(edges) + 1
    else:
        return np.max(edges) + 1


def unique_segment(data, dtype="int64"):
    """Return Segment Id from data
    """
    unique, index = paddle.unique(data, return_inverse=True, dtype=dtype)
    return unique, index


def graph_send_recv(x, src_index, dst_index, pool_type="sum"):
    """This method combines the send and recv function in different pool_type.

    Now, this method only supports default copy send function, and built-in receive pool_type
    function ('sum', 'mean', 'max', 'min').

    Args:

        x (Tensor): The input tensor, and the available data type is float32, float64, int32, int64.

        src_index (Tensor): An 1-D tensor, and the available data type is int32, int64.

        dst_index (Tensor): An 1-D tensor, and should have the same shape as `src_index`. 
                            The available data type is int32, int64. 

        pool_type (str): The pooling type of graph_send_recv, including `sum`, `mean`, 
                         `max`, `min`. Default value is `sum`.

    Returns:

        out (Tensor): The output tensor, should have the same shape and same dtype as input tensor `x`.

    """

    # TODO:@ZHUI add support for 'mean', 'max', 'min' pool_type.
    assert pool_type == "sum", "Only implement 'sum' pool_type function right now. Maybe you can update PaddlePaddle version to fix this problem."

    def send(message_func, src_feat):
        src_feat_temp = {}
        if src_feat is not None:
            assert isinstance(src_feat,
                              dict), "The input src_feat must be a dict"
            src_feat_temp.update(src_feat)
        src_feat = op.RowReader(src_feat_temp, src_index)
        msg = message_func(src_feat)
        return msg

    def _sum_recv(feat):
        output_dim = feat.shape[-1]
        init_output = paddle.zeros(
            shape=[x.shape[0], output_dim], dtype=feat.dtype)
        final_output = paddle.scatter(
            init_output, dst_index, feat, overwrite=False)
        return final_output

    msg = send(lambda sf: {"msg": sf["h"]}, src_feat={"h": x})

    return eval("_%s_recv" % pool_type)(msg["msg"])
