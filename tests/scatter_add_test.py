# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""scatter test cases"""

import unittest

import numpy as np
import paddle.fluid as fluid


class ScatterAddTest(unittest.TestCase):
    """ScatterAddTest"""

    def test_scatter_add(self):
        """test_scatter_add"""
        with fluid.dygraph.guard(fluid.CPUPlace()):
            input = fluid.dygraph.to_variable(
                np.array(
                    [[1, 2], [5, 6]], dtype='float32'), )
            index = fluid.dygraph.to_variable(np.array([1, 1], dtype=np.int32))
            updates = fluid.dygraph.to_variable(
                np.array(
                    [[3, 4], [3, 4]], dtype='float32'), )
            output = fluid.layers.scatter(input, index, updates, mode='add')
            assert output.numpy().tolist() == [[1, 2], [11, 14]]


if __name__ == '__main__':
    unittest.main()
