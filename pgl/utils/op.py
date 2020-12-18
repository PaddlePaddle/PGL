# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
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
op package provide some common ops for building paddle model.

"""
import paddle
import numpy as np
from pgl.utils.helper import check_is_tensor


def read_rows(data, index):
    """Slice tensor with given index from dictionary of tensor or tensor

    This function helps to slice data from nested dictionary structure.

    Args:
        data: A dictionary of tensor or tensor

        index: A tensor of slicing index

    Returns:
        Return a dictionary of tensor or tensor.
    """
    if data is None:
        return None
    elif isinstance(data, dict):
        new_data = {}
        for key, value in data.items():
            new_data[key] = read_rows(value, index)
        return new_data
    else:
        return paddle.gather(data, index)


def get_index_from_counts(counts):
    """Return index generated from counts

    This function return the index from given counts.

    For example, when counts = [ 2, 3, 4], return [0, 2, 5, 9]

    Args:
        counts: numpy.ndarray of paddle.Tensor

    Return:
        Return idnex of the counts

    """
    if check_is_tensor(counts):
        index = paddle.concat(
            [paddle.zeros(
                shape=[1, ], dtype="int64"), paddle.cumsum(counts)],
            axis=-1)
    else:
        index = np.cumsum(counts, dtype="int64")
        index = np.insert(index, 0, 0)
    return index


class RowReader(dict):
    """Memory Efficient RowReader 
    """

    def __init__(self, nfeat, index):
        self.nfeat = nfeat
        self.loaded_nfeat = {}
        self.index = index

    def __getitem__(self, key):
        if key not in self.loaded_nfeat:
            self.loaded_nfeat[key] = read_rows(self.nfeat[key], self.index)
        return self.loaded_nfeat[key]
