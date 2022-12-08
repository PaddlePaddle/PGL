# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import pgl


class Gin(nn.Layer):
    def __init__(self, hidden_size, act):
        super(Gin, self).__init__()
        self.lin = nn.Linear(hidden_size, hidden_size)
        self.act = act

    def forward(self, graph, x, next_num_nodes):
        src, dst = graph.edges[:, 0], graph.edges[:, 1]
        neigh_feature = paddle.geometric.send_u_recv(
            x, src, dst, pool_type="sum", out_size=next_num_nodes)
        self_feature = x[:next_num_nodes]
        output = self_feature + neigh_feature
        output = self.lin(output)
        if self.act is not None:
            output = getattr(F, self.act)(output)
        output = output + self_feature
        return output


class GraphSageMean(nn.Layer):
    def __init__(self, hidden_size, act):
        super(GraphSageMean, self).__init__()
        self.lin = nn.Linear(2 * hidden_size, hidden_size)
        self.act = act

    def forward(self, graph, x, next_num_nodes):
        src, dst = graph.edges[:, 0], graph.edges[:, 1]
        neigh_feature = paddle.geometric.send_u_recv(
            x, src, dst, pool_type="mean", out_size=next_num_nodes)
        self_feature = x[:next_num_nodes]
        output = paddle.concat([self_feature, neigh_feature], axis=1)
        output = self.lin(output)
        if self.act is not None:
            output = getattr(F, self.act)(output)
        output = F.normalize(output, axis=-1)
        return output


class GraphSageBow(nn.Layer):
    def __init__(self, hidden_size, act):
        super(GraphSageBow, self).__init__()

    def forward(self, graph, x, next_num_nodes):
        src, dst = graph.edges[:, 0], graph.edges[:, 1]
        neigh_feature = paddle.geometric.send_u_recv(
            x, src, dst, pool_type="mean", out_size=next_num_nodes)
        self_feature = x[:next_num_nodes]
        output = self_feature + neigh_feature
        output = F.normalize(output, axis=-1)
        return output


class GraphSageMax(nn.Layer):
    def __init__(self, hidden_size, act):
        super(GraphSageMax, self).__init__()
        self.lin = nn.Linear(2 * hidden_size, hidden_size)
        self.act = act

    def forward(self, graph, x, next_num_nodes):
        src, dst = graph.edges[:, 0], graph.edges[:, 1]
        neigh_feature = paddle.geometric.send_u_recv(
            x, src, dst, pool_type="max", out_size=next_num_nodes)
        self_feature = x[:next_num_nodes]
        output = paddle.concat([self_feature, neigh_feature], axis=1)
        output = self.lin(output)
        if self.act is not None:
            output = getattr(F, self.act)(output)
        output = F.normalize(output, axis=-1)
        return output


class Gat(nn.Layer):
    def __init__(self, hidden_size, act):
        super(Gat, self).__init__()
        self.gnn = pgl.nn.GATConv(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_heads=1,
            feat_drop=0,
            attn_drop=0,
            activation=act)
        self.lin = nn.Linear(hidden_size * 2, hidden_size)
        self.act = act

    def forward(self, graph, x, next_num_nodes):
        neigh_feature = self.gnn(graph, x)[:next_num_nodes]
        self_feature = x[:next_num_nodes]
        output = self.lin(paddle.concat([self_feature, neigh_feature], axis=1))
        if self.act is not None:
            output = getattr(F, self.act)(output)
        return output


class LightGcn(nn.Layer):
    def __init__(self, hidden_size, act):
        super(LightGcn, self).__init__()

    def forward(self, graph, x, next_num_nodes):
        src, dst = graph.edges[:, 0], graph.edges[:, 1]
        neigh_feature = paddle.geometric.send_u_recv(
            x, src, dst, pool_type="sum", out_size=next_num_nodes)
        return neigh_feature
