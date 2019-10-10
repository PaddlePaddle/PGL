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
    Comment.
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import unittest

import paddle.fluid as F
import paddle.fluid.layers as L

from pgl.layers.set2set import Set2Set


def paddle_easy_run(model_func, data):
    prog = F.Program()
    startup_prog = F.Program()
    with F.program_guard(prog, startup_prog):
        ret = model_func()
    place = F.CUDAPlace(0)
    exe = F.Executor(place)
    exe.run(startup_prog)
    return exe.run(prog, fetch_list=ret, feed=data)


class Set2SetTest(unittest.TestCase):
    """Set2SetTest
    """

    def test_graphsage_sample(self):
        """test_graphsage_sample
        """
        import numpy as np

        def model_func():
            s2s = Set2Set(5, 1, 3)
            h0 = L.data(
                name='h0',
                shape=[2, 10, 5],
                dtype='float32',
                append_batch_size=False)
            h1 = s2s.forward(h0)
            return h1,

        data = {"h0": np.random.rand(2, 10, 5).astype("float32")}
        h1, = paddle_easy_run(model_func, data)

        self.assertEqual(h1.shape[0], 2)
        self.assertEqual(h1.shape[1], 10)


if __name__ == "__main__":
    unittest.main()
