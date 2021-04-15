# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# # Licensed under the Apache License, Version 2.0 (the "License");
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

import pgl
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import paddle.fluid.layers as L
from pgl.utils.logger import log
from .gat_conv import linear_init, GATConv


class GNNModel(nn.Layer):
    """Implement of GAT
    """

    def __init__(self,
                 input_size,
                 num_class,
                 num_layers=1,
                 feat_drop=0.6,
                 attn_drop=0.6,
                 num_heads=8,
                 hidden_size=8,
                 drop=0.1,
                 edge_type=5,
                 **kwargs):
        super(GNNModel, self).__init__()
        self.num_class = num_class
        self.num_layers = num_layers
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.drop = drop
        self.edge_type = edge_type

        self.gats = nn.LayerList()
        self.skips = nn.LayerList()
        self.norms = nn.LayerList()

        fc_w_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(1.0))
        fc_bias_attr = paddle.ParamAttr(
            initializer=nn.initializer.Constant(0.0))

        for i in range(self.num_layers):
            self.norms.append(
                nn.BatchNorm1D(
                    self.hidden_size,
                    momentum=0.9,
                    weight_attr=fc_w_attr,
                    bias_attr=fc_bias_attr,
                    data_format='NC'))
            if i == 0:
                self.skips.append(
                    linear_init(
                        input_size, self.hidden_size, init_type='linear'))
                self.gats.append(
                    nn.LayerList([
                        GATConv(
                            input_size,
                            self.hidden_size // self.num_heads,
                            self.feat_drop,
                            self.attn_drop,
                            self.num_heads,
                            activation=None) for _ in range(edge_type)
                    ]))
            else:
                self.skips.append(
                    linear_init(
                        self.hidden_size, self.hidden_size,
                        init_type='linear'))
                self.gats.append(
                    nn.LayerList([
                        GATConv(
                            self.hidden_size,
                            self.hidden_size // self.num_heads,
                            self.feat_drop,
                            self.attn_drop,
                            self.num_heads,
                            activation=None) for _ in range(edge_type)
                    ]))

        self.mlp = nn.Sequential(
            linear_init(
                self.hidden_size, self.hidden_size, init_type='linear'),
            nn.BatchNorm1D(
                self.hidden_size,
                momentum=0.9,
                weight_attr=fc_w_attr,
                bias_attr=fc_bias_attr,
                data_format='NC'),
            nn.ReLU(),
            nn.Dropout(p=self.drop),
            linear_init(
                self.hidden_size, self.num_class, init_type='linear'), )

        self.dropout = nn.Dropout(p=self.drop)

    def get_subgraph_by_masked(self, graph, mask):
        index = L.where(mask)
        if index.shape[0] > 0:
            edges = graph.edges
            sub_edges = paddle.gather(edges, index, axis=0)
            sg = pgl.Graph(sub_edges, num_nodes=graph.num_nodes)
            return sg
        else:
            return None

    def forward(self, graph_list, feature):
        for idx, (sg, sub_index) in enumerate(graph_list):
            #feature = paddle.gather(feature, sub_index, axis=0)
            skip_feat = paddle.gather(feature, sub_index, axis=0)
            skip_feat = self.skips[idx](skip_feat)

            for i in range(self.edge_type):
                masked = sg.edge_feat['edge_type'] == i
                m_sg = self.get_subgraph_by_masked(sg, masked)
                if m_sg is not None:
                    feature_temp = self.gats[idx][i](m_sg, feature)
                    feature_temp = paddle.gather(
                        feature_temp, sub_index, axis=0)
                    skip_feat += feature_temp

            feature = F.elu(self.norms[idx](skip_feat))
            feature = self.dropout(feature)
        output = self.mlp(feature)
        return output
