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
from paddle.common_ops_import import Variable

from pgl.utils import op


def check_is_tensor(*data):
    """Check if the given datas have paddle.Tensor
    """
    for d in data:
        if isinstance(d, paddle.Tensor) or isinstance(d, Variable):
            return True
    return False


def scatter(x, index, updates, overwrite=True, name=None):
    """
    **Scatter Layer**
    Output is obtained by updating the input on selected indices based on updates.
    
    .. code-block:: python
    
        import numpy as np
        #input:
        x = np.array([[1, 1], [2, 2], [3, 3]])
        index = np.array([2, 1, 0, 1])
        # shape of updates should be the same as x
        # shape of updates with dim > 1 should be the same as input
        updates = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        overwrite = False
        # calculation:
        if not overwrite:
            for i in range(len(index)):
                x[index[i]] = np.zeros((2))
        for i in range(len(index)):
            if (overwrite):
                x[index[i]] = updates[i]
            else:
                x[index[i]] += updates[i]
        # output:
        out = np.array([[3, 3], [6, 6], [1, 1]])
        out.shape # [3, 2]
    **NOTICE**: The order in which updates are applied is nondeterministic, 
    so the output will be nondeterministic if index contains duplicates.
    Args:
        x (Tensor): The input N-D Tensor with ndim>=1. Data type can be float32, float64.
        index (Tensor): The index 1-D Tensor. Data type can be int32, int64. The length of index cannot exceed updates's length, and the value in index cannot exceed input's length.
        updates (Tensor): update input with updates parameter based on index. shape should be the same as input, and dim value with dim > 1 should be the same as input.
        overwrite (bool): The mode that updating the output when there are same indices. 
          If True, use the overwrite mode to update the output of the same index,
          if False, use the accumulate mode to update the output of the same index.Default value is True.
        name(str, optional): The default value is None. Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name` .
 
    Returns:
        Tensor: The output is a Tensor with the same shape as x.
    Examples:
        .. code-block:: python
            
            import paddle
            x = paddle.to_tensor([[1, 1], [2, 2], [3, 3]], dtype='float32')
            index = paddle.to_tensor([2, 1, 0, 1], dtype='int64')
            updates = paddle.to_tensor([[1, 1], [2, 2], [3, 3], [4, 4]], dtype='float32')
  
            output1 = paddle.scatter(x, index, updates, overwrite=False)
            # [[3., 3.],
            #  [6., 6.],
            #  [1., 1.]]
            output2 = paddle.scatter(x, index, updates, overwrite=True)
            # CPU device:
            # [[3., 3.],
            #  [4., 4.],
            #  [1., 1.]]
            # GPU device maybe have two results because of the repeated numbers in index
            # result 1:
            # [[3., 3.],
            #  [4., 4.],
            #  [1., 1.]]
            # result 2:
            # [[3., 3.],
            #  [2., 2.],
            #  [1., 1.]]
    """
    return paddle.scatter(x, index, updates, overwrite, name)


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
