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

global_feat_dim = 4


def get_conv_list():
    return [
        pgl.nn.GCNConv(
            input_size=global_feat_dim, output_size=global_feat_dim),
        pgl.nn.GraphSageConv(
            input_size=global_feat_dim, hidden_size=global_feat_dim),
        # pgl.nn.PinSageConv(
        #     input_size=global_feat_dim, hidden_size=global_feat_dim),
        pgl.nn.GATConv(
            input_size=global_feat_dim, hidden_size=global_feat_dim),
        pgl.nn.GCNII(hidden_size=global_feat_dim),
        pgl.nn.APPNP(),
        pgl.nn.SGCConv(
            input_size=global_feat_dim, output_size=global_feat_dim),
        pgl.nn.SSGCConv(
            input_size=global_feat_dim, output_size=global_feat_dim),
    ]


class ConvTest(unittest.TestCase):
    def run_graph_conv(self, dtype="float32"):
        num_nodes = 5

        edges = [(0, 1), (1, 2), (3, 4)]
        nfeat = np.random.randn(num_nodes, global_feat_dim).astype(dtype)
        efeat = np.random.randn(len(edges), global_feat_dim).astype(dtype)
        weight = np.random.randn(len(edges), global_feat_dim).astype(dtype)

        g = pgl.Graph(
            edges=edges,
            num_nodes=num_nodes,
            node_feat={'nfeat': nfeat},
            edge_feat={'efeat': efeat,
                       'weight': weight})
        pg = g.tensor()
        #feat = paddle.to_tensor(nfeat)
        feat = g.node_feat["nfeat"]
        for conv in get_conv_list():
            out = conv(pg, feat)
            self.assertTrue(isinstance(out, paddle.Tensor))

    def test_graph_conv_float32(self):
        paddle.set_default_dtype("float32")
        self.run_graph_conv("float32")

    def test_graph_conv_float64(self):
        paddle.set_default_dtype("float64")
        self.run_graph_conv("float64")


if __name__ == "__main__":
    unittest.main()
