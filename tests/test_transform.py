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
from pgl.utils.transform import to_dense_batch, filter_adj


class TransformTest(unittest.TestCase):
    """TransformTest
    """

    def test_to_dense_batch(self):

        data1 = paddle.to_tensor(
            np.array(
                [[1, 2, 3], [3, 2, 1], [4, 5, 6]], dtype="float32"))
        data2 = paddle.to_tensor(
            np.array(
                [[1, 2, 3], [3, 2, 1]], dtype="float32"))
        data3 = paddle.to_tensor(np.array([[1, 2, 3]], dtype="float32"))
        data = paddle.concat([data1, data2, data3], axis=0)
        common_edges = [(0, 0)]
        graph = pgl.Graph.disjoint(
            [pgl.Graph(
                num_nodes=i, edges=common_edges) for i in [3, 2, 1]],
            merged_graph_index=False).tensor()
        out, mask = to_dense_batch(data, graph, fill_value=0)
        _out = [[[1, 2, 3], [3, 2, 1], [4, 5, 6]],
                [[1, 2, 3], [3, 2, 1], [0, 0, 0]],
                [[1, 2, 3], [0, 0, 0], [0, 0, 0]]]
        self.assertAlmostEqual(out.numpy().tolist(), _out)
        _mask = [[False, False, False], [False, False, True],
                 [False, True, True]]
        self.assertAlmostEqual(mask.numpy().tolist(), _mask)

    def test_filter_adj(self):
        edge_list = [[2, 0], [2, 1], [3, 1], [4, 0], [5, 0], [6, 0], [6, 4],
                     [6, 5], [7, 0], [7, 1], [7, 2], [7, 3], [8, 0], [9, 7]]
        edge_index = paddle.to_tensor(edge_list)
        perm = paddle.to_tensor([0, 1, 2, 4, 5, 8, 9])
        edge_index, _ = filter_adj(edge_index, perm)
        edge_index_ = [[2, 0], [2, 1], [3, 0], [4, 0], [5, 0]]
        self.assertAlmostEqual(edge_index.numpy().tolist(), edge_index_)


if __name__ == "__main__":
    unittest.main()
