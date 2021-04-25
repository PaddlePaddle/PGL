#-*- coding: utf-8 -*-
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


if __name__ == "__main__":
    unittest.main()
