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
    'segment_sum', 'segment_mean', 'segment_max', 'segment_min',
    'segment_softmax'
]

import paddle

from paddle.fluid.framework import core, in_dygraph_mode
from paddle.fluid.layer_helper import LayerHelper, in_dygraph_mode
from paddle.fluid.data_feeder import check_variable_and_dtype


def segment_pool(data, segment_ids, pool_type, name=None):
    """
    Segment Operator.
    """
    pool_type = pool_type.upper()
    if in_dygraph_mode():
        out, tmp = core.ops.segment_pool(data, segment_ids, 'pooltype',
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
    Segment Sum Operator.

    This operator sums the elements of input `data` which with
    the same index in `segment_ids`.
    It computes a tensor such that $out_i = \\sum_{j} data_{j}$
    where sum is over j such that `segment_ids[j] == i`.

    Args:
        data (Tensor): A tensor, available data type float32, float64.
        segment_ids (Tensor): A 1-D tensor, which have the same size
                            with the first dimension of input data. 
                            Available data type is int32, int64.
    Returns:
       output (Tensor): the reduced result.

    Examples:

        .. code-block:: python

            import paddle
            import pgl
            data = paddle.to_tensor([[1, 2, 3], [3, 2, 1], [4, 5, 6]], dtype='float32')
            segment_ids = paddle.to_tensor([0, 0, 1], dtype='int32')
            out = pgl.math.segment_sum(data, segment_ids)
            #Outputs: [[4., 4., 4.], [4., 5., 6.]]

    """
    if in_dygraph_mode():
        out, tmp = core.ops.segment_pool(data, segment_ids, 'pooltype', "SUM")
        return out

    check_variable_and_dtype(data, "X", ("float32", "float64"), "segment_pool")
    check_variable_and_dtype(segment_ids, "SegmentIds", ("int32", "int64"),
                             "segment_pool")

    helper = LayerHelper("segment_sum", **locals())
    out = helper.create_variable_for_type_inference(dtype=data.dtype)
    summed_ids = helper.create_variable_for_type_inference(dtype=data.dtype)
    helper.append_op(
        type="segment_pool",
        inputs={"X": data,
                "SegmentIds": segment_ids},
        outputs={"Out": out,
                 "SummedIds": summed_ids},
        attrs={"pooltype": "SUM"})
    return out


def segment_mean(data, segment_ids, name=None):
    """
    Segment mean Operator.

    Ihis operator calculate the mean value of input `data` which
    with the same index in `segment_ids`.
    It computes a tensor such that $out_i = \\frac{1}{n_i}  \\sum_{j} data[j]$
    where sum is over j such that 'segment_ids[j] == i' and $n_i$ is the number
    of all index 'segment_ids[j] == i'.

    Args:
        data (tensor): a tensor, available data type float32, float64.
        segment_ids (tensor): a 1-d tensor, which have the same size 
                            with the first dimension of input data. 
                            available data type is int32, int64.

    Returns:
       output (Tensor): the reduced result.

    Examples:

        .. code-block:: python

            import paddle
            import pgl
            data = paddle.to_tensor([[1, 2, 3], [3, 2, 1], [4, 5, 6]], dtype='float32')
            segment_ids = paddle.to_tensor([0, 0, 1], dtype='int32')
            out = pgl.math.segment_mean(data, segment_ids)
            #Outputs: [[2., 2., 2.], [4., 5., 6.]]

    """
    if in_dygraph_mode():
        out, tmp = core.ops.segment_pool(data, segment_ids, 'pooltype', "MEAN")
        return out

    check_variable_and_dtype(data, "X", ("float32", "float64"), "segment_pool")
    check_variable_and_dtype(segment_ids, "SegmentIds", ("int32", "int64"),
                             "segment_pool")

    helper = LayerHelper("segment_mean", **locals())
    out = helper.create_variable_for_type_inference(dtype=data.dtype)
    summed_ids = helper.create_variable_for_type_inference(dtype=data.dtype)
    helper.append_op(
        type="segment_pool",
        inputs={"X": data,
                "SegmentIds": segment_ids},
        outputs={"Out": out,
                 "SummedIds": summed_ids},
        attrs={"pooltype": "MEAN"})
    return out


def segment_min(data, segment_ids, name=None):
    """
    Segment min operator.

    This operator calculate the minimum elements of input `data` which with
    the same index in `segment_ids`.
    It computes a tensor such that $out_i = \\min_{j} data_{j}$
    where min is over j such that `segment_ids[j] == i`.

    Args:
        data (tensor): a tensor, available data type float32, float64.
        segment_ids (tensor): a 1-d tensor, which have the same size
                            with the first dimension of input data. 
                            available data type is int32, int64.
    Returns:
       output (Tensor): the reduced result.

    Examples:

        .. code-block:: python

            import paddle
            import pgl
            data = paddle.to_tensor([[1, 2, 3], [3, 2, 1], [4, 5, 6]], dtype='float32')
            segment_ids = paddle.to_tensor([0, 0, 1], dtype='int32')
            out = pgl.math.segment_min(data, segment_ids)
            #Outputs:  [[1., 2., 1.], [4., 5., 6.]]

    """
    if in_dygraph_mode():
        out, tmp = core.ops.segment_pool(data, segment_ids, 'pooltype', "MIN")
        return out

    check_variable_and_dtype(data, "X", ("float32", "float64"), "segment_pool")
    check_variable_and_dtype(segment_ids, "SegmentIds", ("int32", "int64"),
                             "segment_pool")

    helper = LayerHelper("segment_min", **locals())
    out = helper.create_variable_for_type_inference(dtype=data.dtype)
    summed_ids = helper.create_variable_for_type_inference(dtype=data.dtype)
    helper.append_op(
        type="segment_pool",
        inputs={"X": data,
                "SegmentIds": segment_ids},
        outputs={"Out": out,
                 "SummedIds": summed_ids},
        attrs={"pooltype": "MIN"})
    return out


def segment_max(data, segment_ids, name=None):
    """
    Segment max operator.

    This operator calculate the maximum elements of input `data` which with
    the same index in `segment_ids`.
    It computes a tensor such that $out_i = \\min_{j} data_{j}$
    where max is over j such that `segment_ids[j] == i`.

    Args:
        data (tensor): a tensor, available data type float32, float64.
        segment_ids (tensor): a 1-d tensor, which have the same size
                            with the first dimension of input data. 
                            available data type is int32, int64.

    Returns:
       output (Tensor): the reduced result.

    Examples:

        .. code-block:: python

            import paddle
            import pgl
            data = paddle.to_tensor([[1, 2, 3], [3, 2, 1], [4, 5, 6]], dtype='float32')
            segment_ids = paddle.to_tensor([0, 0, 1], dtype='int32')
            out = pgl.math.segment_max(data, segment_ids)
            #Outputs: [[3., 2., 3.], [4., 5., 6.]]

    """
    if in_dygraph_mode():
        out, tmp = core.ops.segment_pool(data, segment_ids, 'pooltype', "MAX")
        return out

    check_variable_and_dtype(data, "X", ("float32", "float64"), "segment_pool")
    check_variable_and_dtype(segment_ids, "SegmentIds", ("int32", "int64"),
                             "segment_pool")

    helper = LayerHelper("segment_max", **locals())
    out = helper.create_variable_for_type_inference(dtype=data.dtype)
    summed_ids = helper.create_variable_for_type_inference(dtype=data.dtype)
    helper.append_op(
        type="segment_pool",
        inputs={"X": data,
                "SegmentIds": segment_ids},
        outputs={"Out": out,
                 "SummedIds": summed_ids},
        attrs={"pooltype": "MAX"})
    return out


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
            data = paddle.to_tensor([[1, 2, 3], [3, 2, 1], [4, 5, 6]], dtype='float32')
            segment_ids = paddle.to_tensor([0, 0, 1], dtype='int32')
            out = pgl.math.segment_softmax(data, segment_ids)
            #Outputs: [[0.11920292, 0.50000000, 0.88079703], [0.88079709, 0.50000000, 0.11920292], [1., 1., 1.]]
    
    """
    data_max = segment_max(data, segment_ids)
    data_max = paddle.gather(data, segment_ids, axis=0)
    data = data - data_max
    data = paddle.exp(data)
    sum_data = segment_sum(data, segment_ids)
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
        max_padding: the max number of elements grouped by same segment_ids
     
    Examples:
    
        .. code-block:: python
    
            import paddle
            import pgl
            data = paddle.to_tensor([[1, 2, 3], [3, 2, 1], [4, 5, 6]], dtype='float32')
            segment_ids = paddle.to_tensor([0, 0, 1], dtype='int64')
            out = pgl.math.segment_softmax(data, segment_ids)
            #Outputs: [[[1., 2., 3.], [3., 2., 1.]], [[4., 5., 6.], [0., 0., 0.]]], [2,1], 2

    """
    idx_a = segment_ids
    idx_b = paddle.arange(segment_ids.shape[0])
    
    temp_idx  = paddle.ones([segment_ids.shape[0]], dtype='float32')
    temp_idx = segment_sum(temp_idx, segment_ids).astype('int32')
    
    seq_len = temp_idx
    max_padding = temp_idx.max().numpy()[0]
    
    temp_idx = paddle.cumsum(temp_idx)
    temp_idx_i = paddle.zeros([temp_idx.shape[0] + 1], dtype='int32')
    temp_idx_i[1: ] = temp_idx
    temp_idx = temp_idx_i[: -1]
    temp_idx = paddle.gather(temp_idx, segment_ids)
    
    idx_b = idx_b - temp_idx
    index = paddle.stack([idx_a, idx_b], axis=1)
    
    bz = segment_ids.max().numpy()[0] + 1
    
    shape = [bz, max_padding, data.shape[-1]]
    output = paddle.scatter_nd(index, data, shape)
    
    return output, seq_len, index

