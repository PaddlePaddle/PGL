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

import paddle
import pgl

import paddle.nn as nn
import pgl.nn as gnn
from ogb.utils import smiles2graph

graph_obj = smiles2graph('O=C1C=CC(O1)C(c1ccccc1C)O')


class GNNModel(nn.Layer):
    def __init__(self, input_size, output_size, num_layers=3):
        super(GNNModel, self).__init__()
        self.conv_fn = nn.LayerList()
        self.conv_fn.append(gnn.GCNConv(input_size, output_size))
        for i in range(num_layers - 1):
            self.conv_fn.append(gnn.GCNConv(output_size, output_size))
        self.pool_fn = gnn.GraphPool("sum")

    def forward(self, graph, feature):
        for fn in self.conv_fn:
            feature = fn(graph, feature)
        output = self.pool_fn(graph, feature)
        return output


graph = pgl.Graph(
    num_nodes=len(graph_obj["node_feat"]),
    edges=graph_obj["edge_index"].T,
    node_feat={"feat": graph_obj["node_feat"].astype("float32")})
graph.tensor()
model = GNNModel(graph_obj["node_feat"].shape[-1], 10)
output = model(graph, graph.node_feat["feat"])
print(output.numpy())
paddle.save(model.state_dict(), "gnn.pdparam")
