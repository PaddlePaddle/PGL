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

__all__ = [
    'segment_pool',
    'segment_sum',
    'segment_mean',
    'segment_max',
    'segment_min',
    'segment_softmax',
    'segment_padding',
]

import paddle
from paddle import _C_ops
from paddle.fluid.framework import core, in_dygraph_mode
from paddle.fluid.layer_helper import LayerHelper, in_dygraph_mode
from paddle.fluid.data_feeder import check_variable_and_dtype

from pgl.utils.op import get_index_from_counts


def segment_pool(data, segment_ids, pool_type, name=None):
    """
    Segment Operator.
    """
    pool_type = pool_type.upper()
    if in_dygraph_mode():
        out, tmp = _C_ops.segment_pool(data, segment_ids, 'pooltype',
                                       pool_type)
        return out

    check_variable_and_dtype(data, "X", ("float32", "float64"), "segment_pool")
    check_variable_and_dtype(segment_ids, "SegmentIds", ("int32", "int64"),
                             "segment_pool")

    helper = LayerHelper("segment_pool", **locals())
    out = helper.create_variable_for_type_inference(dtype=data.dtype)
    pool_ids = helper.create_variable_for_type_inference(dtype=data.dtype)
    helper.append_op(
        type="segment_pool",
        inputs={"X": data,
                "SegmentIds": segment_ids},
        outputs={"Out": out,
                 "SummedIds": pool_ids},
        attrs={"pooltype": pool_type})
    return out


def segment_sum(data, segment_ids, name=None):
    """
    See note of paddle.incubate.segment_sum
    """
    return paddle.incubate.segment_sum(data, segment_ids, name=name)


def segment_mean(data, segment_ids, name=None):
    """
    See note of paddle.incubate.segment_mean
    """
    return paddle.incubate.segment_mean(data, segment_ids, name=name)


def segment_max(data, segment_ids, name=None):
    """
    See note of paddle.incubate.segment_max
    """
    return paddle.incubate.segment_max(data, segment_ids, name=name)


def segment_min(data, segment_ids, name=None):
    """
    See note of paddle.incubate.segment_min
    """
    return paddle.incubate.segment_min(data, segment_ids, name=name)


def segment_softmax(data, segment_ids):
    """
    Segment softmax operator.
    
    This operator calculate the softmax elements of input `data` which with
    the same index in `segment_ids`.
    
    Args:
        data (tensor): a tensor, available data type float32, float64.
        segment_ids (tensor): a 1-d tensor, which have the same size
                            with the first dimension of input data. 
                            available data type is int32, int64.
    
    Returns:
       output (Tensor): the softmax result.
    
    Examples:
    
        .. code-block:: python
    
            import paddle
            import pgl
            data = [[1, 2, 3], 
                    [3, 2, 1], 
                    [4, 5, 6]]
            data = paddle.to_tensor(data, dtype='float32')
            segment_ids = paddle.to_tensor([0, 0, 1], dtype='int32')
            out = pgl.math.segment_softmax(data, segment_ids)

            # Outputs:
                    [[0.11920292 0.5        0.880797  ]
                     [0.880797   0.5        0.11920292]
                     [1.         1.         1.        ]]

    """
    with paddle.no_grad():
        # no need gradients
        data_max = paddle.incubate.segment_max(data, segment_ids)
        data_max = paddle.gather(data_max, segment_ids, axis=0)
    data = data - data_max
    data = paddle.exp(data)
    sum_data = paddle.incubate.segment_sum(data, segment_ids)
    sum_data = paddle.gather(sum_data, segment_ids, axis=0)
    return data / sum_data


def segment_padding(data, segment_ids):
    """
    Segment padding operator.

    This operator padding the input elements which with the same index in 'segment_ids' to a common length ,
    and reshape its into [uniq_segment_id, max_padding, dim].
    Args:
        data (tensor): a tensor, available data type float32, float64.
        segment_ids (tensor): a 1-d tensor, which have the same size
                            with the first dimension of input data.
                            available data type is int32, int64.

    Returns:
        output (Tensor): the padding result with shape [uniq_segment_id, max_padding, dim].
        seq_len (Tensor): the numbers of elements grouped same segment_ids
        index: The index of elements for gather_nd or scatter_nd operation

    Examples:

        .. code-block:: python

            import paddle
            import pgl
            data = paddle.to_tensor([[1, 2, 3], [3, 2, 1], [4, 5, 6]], dtype='float32')
            segment_ids = paddle.to_tensor([0, 0, 1], dtype='int64')
            output, seq_len, index = pgl.math.segment_padding(data, segment_ids)
    """
    idx_a = segment_ids
    idx_b = paddle.arange(paddle.shape(segment_ids)[0])

    temp_idx = paddle.ones_like(segment_ids, dtype='float32')
    segment_len = paddle.incubate.segment_sum(temp_idx,
                                              segment_ids).astype('int32')

    max_padding = paddle.max(segment_len)

    segment_shift = get_index_from_counts(segment_len)[:-1]
    segment_shift = paddle.gather(segment_shift, segment_ids)

    idx_b = idx_b - segment_shift

    index = paddle.stack([idx_a, idx_b], axis=1)

    shape = [paddle.shape(segment_len)[0], max_padding, data.shape[-1]]
    output = paddle.scatter_nd(index, data, shape)

    return output, segment_len, index
