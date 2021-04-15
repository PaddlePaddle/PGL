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
import paddle.nn as nn
import paddle.nn.functional as F

import pgl.nn as gnn

from conv import GNN_node, GNN_node_Virtualnode


class GNN(paddle.nn.Layer):
    def __init__(self,
                 num_tasks=1,
                 num_layers=5,
                 emb_dim=300,
                 gnn_type='gin',
                 virtual_node=True,
                 residual=False,
                 drop_ratio=0,
                 JK="last",
                 graph_pooling="sum"):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''
        super(GNN, self).__init__()

        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(
                num_layers,
                emb_dim,
                JK=JK,
                drop_ratio=drop_ratio,
                residual=residual,
                gnn_type=gnn_type)
        else:
            self.gnn_node = GNN_node(
                num_layers,
                emb_dim,
                JK=JK,
                drop_ratio=drop_ratio,
                residual=residual,
                gnn_type=gnn_type)

        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = gnn.GraphPool(pool_type="sum")
        elif self.graph_pooling == "mean":
            self.pool = gnn.GraphPool(pool_type="mean")
        elif self.graph_pooling == "max":
            self.pool = gnn.GraphPool(pool_type="max")
        else:
            raise ValueError("Invalid graph pooling type.")

        self.graph_pred_linear = nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, g):
        h_node = self.gnn_node(g)

        h_graph = self.pool(g, h_node)
        output = self.graph_pred_linear(h_graph)

        if self.training:
            return output
        else:
            # At inference time, relu is applied to output to ensure positivity
            return paddle.clip(output, min=0, max=50)


if __name__ == '__main__':
    GNN(num_tasks=10)
