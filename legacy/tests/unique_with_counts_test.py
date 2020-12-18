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
"""unique with counts test"""

import unittest

import numpy as np
import paddle.fluid as fluid


class UniqueWithCountTest(unittest.TestCase):
    """UniqueWithCountTest"""

    def _test_unique_with_counts_helper(self, input, output):
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(main_program, startup_program):
            x = fluid.layers.data(
                name='input',
                dtype='int64',
                shape=[-1],
                append_batch_size=False)
            #x = fluid.assign(np.array([2, 3, 3, 1, 5, 3], dtype='int32'))
            out, index, count = fluid.layers.unique_with_counts(x)

        out, index, count = exe.run(
            main_program,
            feed={'input': np.array(
                input, dtype='int64'), },
            fetch_list=[out, index, count],
            return_numpy=True, )
        out, index, count = out.tolist(), index.tolist(), count.tolist()
        assert [out, index, count] == output

    def test_unique_with_counts(self):
        """test_unique_with_counts"""
        self._test_unique_with_counts_helper(
            input=[1, 1, 2, 4, 4, 4, 7, 8, 8],
            output=[
                [1, 2, 4, 7, 8],
                [0, 0, 1, 2, 2, 2, 3, 4, 4],
                [2, 1, 3, 1, 2],
            ], )
        self._test_unique_with_counts_helper(
            input=[1],
            output=[
                [1],
                [0],
                [1],
            ], )
        self._test_unique_with_counts_helper(
            input=[1, 1],
            output=[
                [1],
                [0, 0],
                [2],
            ], )


if __name__ == '__main__':
    unittest.main()
