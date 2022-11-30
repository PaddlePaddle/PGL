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

import unittest
import numpy as np

import paddle
import pgl
import pgl.nn as nn
import pgl.nn.functional as F


class GraphOpTest(unittest.TestCase):
    def test_graph_norm(self):
        graph_list = []

        edges1 = [(0, 1), (1, 2)]
        num_nodes1 = 3
        g1 = pgl.Graph(edges=edges1, num_nodes=num_nodes1)
        graph_list.append(g1)

        edges2 = [(0, 2), (0, 3), (1, 2)]
        num_nodes2 = 4
        g2 = pgl.Graph(edges=edges2, num_nodes=num_nodes2)
        graph_list.append(g2)

        multi_graph = pgl.Graph.disjoint(graph_list)
        multi_graph.tensor()

        feat = np.repeat(
            np.arange(0, 7).reshape(-1, 1), 3, axis=1).astype("float32")
        tensor_feat = paddle.to_tensor(feat, dtype="float32")

        feat[0:3] = feat[0:3] / np.sqrt(3)
        feat[3:] = feat[3:] / np.sqrt(4)

        norm_feat = F.graph_norm(multi_graph, tensor_feat)
        self.assertEqual(feat.tolist(), norm_feat.numpy().tolist())

        gn_layer = nn.GraphNorm()
        norm_feat = gn_layer(multi_graph, tensor_feat)

        self.assertEqual(feat.tolist(), norm_feat.numpy().tolist())

    def test_edge_softmax(self):
        num_nodes = 3
        edges = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        g = pgl.Graph(edges=edges, num_nodes=num_nodes)
        g.tensor()
        logits = paddle.to_tensor([1, 1, 1, 1, 1, 1], dtype="float32")
        softmax_res = F.edge_softmax(g, logits, norm_by="dst")
        res = np.array([1., 0.5, 1 / 3, 0.5, 1 / 3, 1 / 3], dtype="float32")
        self.assertTrue((softmax_res.numpy() == res).all())

        softmax_res2 = F.edge_softmax(g, logits, norm_by="src")
        res2 = np.array([1 / 3, 1 / 3, 1 / 3, 0.5, 0.5, 1.], dtype="float32")
        self.assertTrue((softmax_res2.numpy() == res2).all())


if __name__ == "__main__":
    unittest.main()
