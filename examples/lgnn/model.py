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

import copy
import itertools
import pgl
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import paddle


def aggregate_radius(radius, graph, feature):
    feat_list = []
    feature = graph.send_recv(feature, "sum")
    feat_list.append(feature)
    for i in range(radius - 1):
        for j in range(2**i):
            feature = graph.send_recv(feature, "sum")
        feat_list.append(feature)
    return feat_list


class LGNNCore(nn.Layer):
    def __init__(self, in_feats, out_feats, radius):
        super(LGNNCore, self).__init__()
        self.out_feats = out_feats
        self.radius = radius

        self.linear_prev = nn.Linear(in_feats, out_feats)
        self.linear_deg = nn.Linear(in_feats, out_feats)
        self.linear_radius = nn.LayerList(
            [nn.Linear(in_feats, out_feats) for i in range(radius)])
        # self.linear_fuse = nn.Linear(in_feats, out_feats)
        self.bn = nn.BatchNorm1D(out_feats)

    def forward(self, graph, feat_a, feat_b, deg):
        # term "prev"
        prev_proj = self.linear_prev(feat_a)
        # term "deg"
        deg_proj = self.linear_deg(deg * feat_a)
        # term "radius" "aggregate 2^j-hop features
        hop2j_list = aggregate_radius(self.radius, graph, feat_a)
        # apply linear transformation
        hop2j_list = [
            linear(x) for linear, x in zip(self.linear_radius, hop2j_list)
        ]
        radius_proj = sum(hop2j_list)

        # TODO add fuse
        # sum them together
        result = prev_proj + deg_proj + radius_proj

        # skip connection and batch norm
        n = self.out_feats // 2
        result = paddle.concat([result[:, :n], F.relu(result[:, n:])], 1)
        result = self.bn(result)
        return result


class LGNNLayer(nn.Layer):
    def __init__(self, in_feats, out_feats, radius):
        super(LGNNLayer, self).__init__()
        self.g_layer = LGNNCore(in_feats, out_feats, radius)
        self.lg_layer = LGNNCore(in_feats, out_feats, radius)

    def forward(self, graph, line_graph, feat, lg_feat, deg_g, deg_lg):
        next_feat = self.g_layer(graph, feat, lg_feat, deg_g)
        next_lg_feat = self.lg_layer(line_graph, lg_feat, feat, deg_lg)
        return next_feat, next_lg_feat


class LGNN(nn.Layer):
    def __init__(self, radius):
        super(LGNN, self).__init__()
        self.layer1 = LGNNLayer(1, 16, radius)  # input is scalar feature
        self.layer2 = LGNNLayer(16, 16, radius)  # hidden size is 16
        self.layer3 = LGNNLayer(16, 16, radius)
        self.linear = nn.Linear(16, 2)  # predice two classes

    def forward(self, graph, line_graph):
        # compute the degrees
        deg_g = graph.indegree().astype("float32").unsqueeze(-1)
        #print("deg_g", deg_g)
        deg_lg = line_graph.indegree().astype("float32").unsqueeze(-1)
        #print("deg_lg", deg_lg)
        # use degree as the input feature
        feat, lg_feat = deg_g, deg_lg
        feat, lg_feat = self.layer1(graph, line_graph, feat, lg_feat, deg_g,
                                    deg_lg)
        feat, lg_feat = self.layer2(graph, line_graph, feat, lg_feat, deg_g,
                                    deg_lg)
        feat, lg_feat = self.layer3(graph, line_graph, feat, lg_feat, deg_g,
                                    deg_lg)
        return self.linear(feat)


if __name__ == "__main__":
    g = LGNN(3)
