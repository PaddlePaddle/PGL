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
    'segment_topk',
]

import paddle
from pgl.utils.op import get_index_from_counts


def segment_pool(data, segment_ids, pool_type, name=None):
    """
    Segment Operator.
    """
    pool_type = pool_type.upper()
    if pool_type == "SUM":
        return paddle.geometric.segment_sum(data, segment_ids, name)
    elif pool_type == "MEAN":
        return paddle.geometric.segment_mean(data, segment_ids, name)
    elif pool_type == "MAX":
        return paddle.geometric.segment_max(data, segment_ids, name)
    elif pool_type == "MIN":
        return paddle.geometric.segment_min(data, segment_ids, name)
    else:
        raise ValueError(
            "We only support sum, mean, max, min pool types in segment_pool function."
        )


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

    return paddle.geometric.segment_sum(data, segment_ids, name)


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
    return paddle.geometric.segment_mean(data, segment_ids, name)


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
    return paddle.geometric.segment_min(data, segment_ids, name)


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
    return paddle.geometric.segment_max(data, segment_ids, name)


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
            data = paddle.to_tensor(, dtype='float32')
            segment_ids = paddle.to_tensor([0, 0, 1], dtype='int32')
            out = pgl.math.segment_softmax(data, segment_ids)

            # Outputs:
                    [[0.11920292 0.5        0.880797  ]
                     [0.880797   0.5        0.11920292]
                     [1.         1.         1.        ]]

    """
    with paddle.no_grad():
        # no need gradients
        data_max = segment_max(data, segment_ids)
        data_max = paddle.gather(data_max, segment_ids, axis=0)
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
    segment_len = segment_sum(temp_idx, segment_ids).astype('int32')

    max_padding = paddle.max(segment_len)

    segment_shift = get_index_from_counts(segment_len)[:-1]
    segment_shift = paddle.gather(segment_shift, segment_ids)

    idx_b = idx_b - segment_shift

    index = paddle.stack([idx_a, idx_b], axis=1)

    shape = [paddle.shape(segment_len)[0], max_padding, data.shape[-1]]
    output = paddle.scatter_nd(index, data, shape)

    return output, segment_len, index


@paddle.no_grad()
def __segment_topk_rank(scores, segment_ids, num_nodes, max_num_nodes):
    """
    Used by segment_topk.
    This function offer the rank results according to scores.
    """
    batch_size = int(num_nodes.shape[0])
    nodes_cumsum = num_nodes.cumsum(0)
    cum_num_nodes = paddle.zeros([1])
    if nodes_cumsum.shape[0] > 1:
        cum_num_nodes = paddle.concat([cum_num_nodes, nodes_cumsum[:-1]], 0)
    index = paddle.arange(segment_ids.shape[0], dtype=paddle.int32)
    index = (index - cum_num_nodes[segment_ids]) + (segment_ids * max_num_nodes
                                                    )
    dense_x = paddle.full([batch_size * max_num_nodes],
                          -1e20).astype(paddle.float32)
    dense_x = paddle.scatter(dense_x, index, scores.reshape([-1]))
    dense_x = dense_x.reshape([batch_size, max_num_nodes])
    perm = dense_x.argsort(-1, descending=True)
    perm = perm + cum_num_nodes.reshape([-1, 1])
    perm = perm.reshape([-1])
    return perm


def segment_topk(x,
                 scores,
                 segment_ids,
                 ratio,
                 min_score=None,
                 return_index=False):
    """
    Segment topk operator.

    This operator select the topk value of input elements which with the same index in 'segment_ids' ,
    and return the index of reserved nodes into with shape [max_num_nodes*ratio, dim].
    
    if min_score is not None, the operator only remove the node with value lower that min_score
    
    Args:
        x (tensor): a tensor, available data type float32, float64.
        segment_ids (tensor): a 1-d tensor, which have the same size
                            with the first dimension of input data.
                            available data type is int32, int64.
        ratio (float): a ratio that reserving present nodes
        min_score (float): if min_score is not None, the operator only remove 
                            the node with value lower than min_score

    Returns:
        perm (Tensor): the index of reserved nodes
    Examples:

        .. code-block:: python

            import paddle
            import pgl
            data = paddle.to_tensor([[1, 2, 3], [3, 2, 1], [4, 5, 6],
                                    [9, 9, 8], [20, 1, 5]], dtype='float32')
            segment_ids = paddle.to_tensor([0, 0, 1, 1, 1], dtype='int64')
            scores = paddle.to_tensor([1, 3, 2, 7, 4])
            output, index = pgl.math.segment_topk(data, scores, segment_ids, 0.5, return_index=True)
    """
    if min_score is not None:
        scores_max = segment_max(scores, segment_ids).index_select(segment_ids,
                                                                   0) - 1e-7
        scores_min = scores_max.clip(max=min_score)
        perm = (scores > scores_min).nonzero(as_tuple=False).reshape([-1])
    else:
        num_nodes = segment_sum(paddle.ones([scores.shape[0]]), segment_ids)
        batch_size, max_num_nodes = int(num_nodes.shape[0]), int(num_nodes.max(
        ).item())
        perm = __segment_topk_rank(scores, segment_ids, num_nodes,
                                   max_num_nodes)
        batch_size = int(num_nodes.shape[0])
        if isinstance(ratio, int):
            k = paddle.full([num_nodes.shape[0]], ratio)
            k = paddle.min(k, num_nodes)
        else:
            k = (ratio *
                 num_nodes.astype(paddle.float32)).ceil().astype(paddle.int64)
        mask = [
            paddle.arange(
                k[i], dtype=paddle.int64) + i * max_num_nodes
            for i in range(batch_size)
        ]
        perm = paddle.concat([perm[i] for i in mask], axis=0)
    out = x[perm]
    if return_index:
        return out, perm
    else:
        return out
