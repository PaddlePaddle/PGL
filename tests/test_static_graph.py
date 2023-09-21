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
import os
import numpy as np

import paddle
import pgl
import paddle.nn as nn
import pgl.nn as gnn
import pgl.nn.functional as F
import paddle.static as static


class GNNModel(nn.Layer):
    def __init__(self, input_size, output_size, num_layers=3):
        super(GNNModel, self).__init__()
        self.conv_fn = nn.LayerList()
        self.conv_fn.append(gnn.GCNConv(input_size, output_size))
        for i in range(num_layers - 1):
            self.conv_fn.append(gnn.GCNConv(output_size, output_size))
        self.pool_fn = gnn.GraphPool("sum")

    def forward(self, num_nodes, edges, feature):
        graph = pgl.Graph(num_nodes=num_nodes, edges=edges)
        for fn in self.conv_fn:
            feature = fn(graph, feature)
        output = self.pool_fn(graph, feature)
        return output


class StaticGraphOpTest(unittest.TestCase):
    def test_static_graph(self):
        path = './tmp'
        dim = 100

        # Load DyGraph Model

        paddle.disable_static()
        num_nodes = 5
        edges = [(0, 1), (1, 2), (3, 4)]
        nfeat = np.random.randn(num_nodes, dim).astype("float32")

        model = GNNModel(dim, 10)
        out = model(
            paddle.to_tensor(num_nodes),
            paddle.to_tensor(edges), paddle.to_tensor(nfeat))
        out = out.numpy()

        paddle.save(model.state_dict(),
                    os.path.join(path, "static_gnn.pdparam"))

        paddle.enable_static()

        # Run Static Fisrt

        model2 = GNNModel(dim, 10)
        input_num_nodes = static.data(
            name='num_nodes', shape=[-1], dtype='int32')
        input_edges = static.data(name='edges', shape=[-1, 2], dtype='int32')
        input_feature = static.data(
            name="feature", shape=[-1, dim], dtype="float32")
        output = model2(input_num_nodes, input_edges, input_feature)

        place = paddle.CPUPlace()
        exe = static.Executor(place)
        exe.run(static.default_startup_program())
        prog = static.default_main_program()

        state_dict = paddle.load(os.path.join(path, "static_gnn.pdparam"))
        model2.set_state_dict(state_dict)

        feed_dict = {
            "num_nodes": np.array([num_nodes]).astype("int32"),
            "edges": np.array(
                edges, dtype="int32"),
            "feature": nfeat.astype("float32"),
        }
        out2 = exe.run(prog, feed=feed_dict, fetch_list=[output])[0]

        eps = np.sum((out2 - out)**2)
        self.assertTrue(eps < 1e-5)

        import shutil
        shutil.rmtree(path)


if __name__ == "__main__":
    unittest.main()
