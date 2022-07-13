# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
import re
import time
import unittest
import numpy as np

import paddle

import pgl
from pgl.utils.logger import log


class MathTest(unittest.TestCase):
    """MathTest
    """

    def test_segment_softmax(self):
        """pgl.math.segment_softmax test
        """
        data = np.array([[1, 2, 3], [3, 2, 1], [4, 5, 6]], dtype="float32")
        seg_ids = np.array([0, 0, 1], dtype="int64")

        ground = np.array(
            [[0.11920292, 0.5, 0.880797], [0.880797, 0.5, 0.11920292],
             [1, 1, 1]],
            dtype="float32")
        ground = ground.reshape(-1, ).tolist()

        data = paddle.to_tensor(data, dtype='float32')
        segment_ids = paddle.to_tensor(seg_ids, dtype='int64')
        out = pgl.math.segment_softmax(data, segment_ids)
        out = out.numpy().reshape(-1, ).tolist()

        self.assertAlmostEqual(out, ground)

        # test nan
        data = np.array(
            [[1, 2, 0.003], [3, 2, 10000000000], [4, 5, 6]], dtype="float32")
        seg_ids = np.array([0, 0, 1], dtype="int64")

        ground = np.array(
            [[0.11920292, 0.5, 0], [0.880797, 0.5, 1], [1, 1, 1]],
            dtype="float32")
        ground = ground.reshape(-1, ).tolist()

        data = paddle.to_tensor(data, dtype='float32')
        segment_ids = paddle.to_tensor(seg_ids, dtype='int64')
        out = pgl.math.segment_softmax(data, segment_ids)
        out = out.numpy().reshape(-1, ).tolist()

        self.assertAlmostEqual(out, ground)

    def test_segment_padding(self):
        """pgl.math.segment_padding test
        """
        data = np.array([[1, 2, 3], [3, 2, 1], [4, 5, 6]], dtype="float32")
        seg_ids = np.array([0, 0, 2], dtype="int64")

        data = paddle.to_tensor(data, dtype='float32')
        segment_ids = paddle.to_tensor(seg_ids, dtype='int64')

        ground = np.array(
            [[[1, 2, 3], [3, 2, 1]], [[0, 0, 0], [0, 0, 0]],
             [[4, 5, 6], [0, 0, 0]]],
            dtype="float32")

        ground = ground.reshape(-1, ).tolist()
        out, _, _ = pgl.math.segment_padding(data, segment_ids)
        out = out.numpy().reshape(-1, ).tolist()
        self.assertAlmostEqual(out, ground)

    def test_segment_topk(self):
        """pgl.math.segment_topk test
        """
        data = paddle.to_tensor(
            [[1, 2, 3], [3, 2, 1], [4, 5, 6], [9, 9, 8], [20, 1, 5]],
            dtype='float32')
        segment_ids = paddle.to_tensor([0, 0, 1, 1, 1], dtype='int64')
        scores = paddle.to_tensor([1, 3, 2, 7, 4], dtype='float32')
        output, index = pgl.math.segment_topk(
            data, scores, segment_ids, 0.5, return_index=True)
        np.testing.assert_almost_equal(index.numpy().tolist(), [1, 3, 4])
        np.testing.assert_almost_equal(output.numpy().tolist(),
                                       [[3, 2, 1], [9, 9, 8], [20, 1, 5]])
        output, index = pgl.math.segment_topk(
            data, scores, segment_ids, 0.5, min_score=2.1, return_index=True)
        np.testing.assert_almost_equal(index.numpy().tolist(), [1, 3, 4])


if __name__ == "__main__":
    unittest.main()
