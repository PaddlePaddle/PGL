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
import math
import paddle

import pgl
from pgl.utils.logger import log
from pgl.math import segment_sum
from pgl.nn import Set2Set, GlobalAttention, SAGPool, GraphPool
from pgl.nn import GraphMultisetTransformer


class PoolTest(unittest.TestCase):
    """PoolTest
    """

    def test_globalattention(self):
        """pgl.nn.GlobalAttention test
        """
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
        gpool = GlobalAttention(paddle.nn.Linear(3, 1))
        output = gpool(graph, data)
        data1 = paddle.to_tensor(
            np.array(
                [[4, 5, 6], [3, 2, 1], [1, 2, 3]], dtype="float32"))
        data2 = paddle.to_tensor(
            np.array(
                [[3, 2, 1], [1, 2, 3]], dtype="float32"))
        data3 = paddle.to_tensor(np.array([[1, 2, 3]], dtype="float32"))
        data = paddle.concat([data1, data2, data3], axis=0)
        output_mix = gpool(graph, data)
        # permutation invariance test
        np.testing.assert_almost_equal(
            output_mix.numpy().reshape(-1, ),
            output.numpy().reshape(-1, ),
            decimal=5, )

    def test_set2set(self):
        """pgl.nn.Set2Set test
        """
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
        set2set = Set2Set(3, 2)
        output = set2set(graph, data)
        data1 = paddle.to_tensor(
            np.array(
                [[4, 5, 6], [3, 2, 1], [1, 2, 3]], dtype="float32"))
        data2 = paddle.to_tensor(
            np.array(
                [[3, 2, 1], [1, 2, 3]], dtype="float32"))
        data3 = paddle.to_tensor(np.array([[1, 2, 3]], dtype="float32"))
        data = paddle.concat([data1, data2, data3], axis=0)
        output_mix = set2set(graph, data)
        # permutation invariance test
        np.testing.assert_almost_equal(
            output_mix.numpy().reshape(-1, ),
            output.numpy().reshape(-1, ),
            decimal=5, )

    def test_gmt(self):
        """pgl.nn.GraphMultisetTransformer test
        """
        num_nodes = 5
        edges = [(0, 1), (1, 2), (3, 4)]
        nfeat = np.random.randn(num_nodes, 12).astype("float32")

        g_1 = pgl.Graph(
            edges=edges, num_nodes=num_nodes, node_feat={'nfeat': nfeat})
        edges = [(0, 1), (1, 2), (3, 4), (5, 2), (0, 5)]
        nfeat = np.random.randn(num_nodes + 1, 12).astype("float32")
        g_2 = pgl.Graph(
            edges=edges, num_nodes=num_nodes + 1, node_feat={'nfeat': nfeat})
        g = pgl.Graph.disjoint([g_1, g_2]).tensor()
        gmt = GraphMultisetTransformer(12, 12, 12, num_nodes=5)
        output = gmt(g, g.node_feat['nfeat'])
        self.assertAlmostEqual(output.shape[0], g.num_graph)

    def test_sag_pool(self):
        """pgl.nn.SAGPool test
        """
        x = paddle.randn((8, 4))
        batch = paddle.to_tensor([0, 0, 1, 1, 1, 2, 2, 2], dtype=paddle.int64)
        edge_index = paddle.to_tensor(
            [[0, 1, 2, 3, 4, 5, 5, 6, 6, 7],
             [1, 0, 3, 4, 2, 6, 7, 7, 5, 6]]).transpose([1, 0])
        num_nodes = segment_sum(paddle.ones([x.shape[0]]), batch)
        batch_size = (batch.max() + 1).item()
        graph_node_index = paddle.zeros([batch_size + 1])
        graph_node_index = paddle.scatter(
            graph_node_index,
            paddle.arange(1, batch_size + 1),
            num_nodes.cumsum(0)).astype(paddle.int64)
        g = pgl.Graph(
            num_nodes=8,
            edges=edge_index,
            node_feat={"attr": x},
            _graph_node_index=graph_node_index,
            _num_graph=batch_size).tensor()
        sag_pool = SAGPool(4, 0.5, pgl.nn.GCNConv)
        output, _, graph = sag_pool(g, x)
        self.assertAlmostEqual(
            output.shape[0],
            math.ceil(0.5 * 2) + math.ceil(0.5 * 3) + math.ceil(0.5 * 3))
        self.assertAlmostEqual(graph.num_graph, 3)
        self.assertAlmostEqual(graph.edges.shape[1], 2)

    def test_mean_pool(self):
        """pgl.nn.GlobalPool test
        """
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
        mean_pool = GraphPool("mean")
        output = mean_pool(graph, data)
        data1 = paddle.to_tensor(
            np.array(
                [[4, 5, 6], [3, 2, 1], [1, 2, 3]], dtype="float32"))
        data2 = paddle.to_tensor(
            np.array(
                [[3, 2, 1], [1, 2, 3]], dtype="float32"))
        data3 = paddle.to_tensor(np.array([[1, 2, 3]], dtype="float32"))
        data = paddle.concat([data1, data2, data3], axis=0)
        output_mix = mean_pool(graph, data)
        # permutation invariance test
        np.testing.assert_almost_equal(
            output_mix.numpy().reshape(-1, ),
            output.numpy().reshape(-1, ),
            decimal=5, )


if __name__ == "__main__":
    unittest.main()
