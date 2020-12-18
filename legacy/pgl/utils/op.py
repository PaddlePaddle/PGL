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
import numpy as np
import paddle.fluid as fluid

from pgl.utils import paddle_helper


def nested_lod_reset(data, reset_index):
    """Reset the lod information as the one in given reset_index.

    This function apply :code:`fluid.layers.lod_reset` recursively
    to all the tensor in nested data.

    Args:
        data: A dictionary of tensor or tensor.
        reset_index: A variable which the target lod information comes from.

    Return:
        Return a dictionary of LodTensor of LodTensor.

    """
    if data is None:
        return None
    elif isinstance(data, dict):
        new_data = {}
        for key, value in data.items():
            new_data[key] = nested_lod_reset(value, reset_index)
        return new_data
    else:
        return fluid.layers.lod_reset(data, reset_index)


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
        return paddle_helper.gather(data, index)


class RowReader(object):
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

