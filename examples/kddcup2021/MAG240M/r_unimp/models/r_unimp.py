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
        self.path_attns = nn.LayerList()
        self.path_attns_linear = nn.LayerList()
        self.path_norms = nn.LayerList()

        self.label_embed = nn.Embedding(num_class, input_size)
        if 'm2v_dim' in kwargs:
            self.m2v_dim = kwargs['m2v_dim']
        else:
            self.m2v_dim = 128
        self.m2v_fc = linear_init(self.m2v_dim, input_size, init_type='linear')


        fc_w_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(1.0))
        fc_bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(0.0))
        for i in range(self.num_layers):
            self.path_attns_linear.append(linear_init(self.hidden_size, self.hidden_size, init_type='linear'))
            #self.path_attns_linear.append(linear_init(self.hidden_size, self.hidden_size, init_type='linear', with_bias=False))
            self.path_attns.append(linear_init(self.hidden_size, 1, init_type='linear'))
            self.path_norms.append(nn.BatchNorm1D(self.hidden_size,
                                             momentum=0.9, weight_attr=fc_w_attr,
                                            bias_attr=fc_bias_attr, data_format='NC'))
            self.norms.append(nn.LayerList([nn.BatchNorm1D(self.hidden_size,
                                             momentum=0.9, weight_attr=fc_w_attr,
                                            bias_attr=fc_bias_attr, data_format='NC') for _ in range(edge_type+1)]))
            if i == 0:
                self.skips.append(linear_init(input_size, self.hidden_size, init_type='linear'))
                self.gats.append(
                    nn.LayerList(
                    [
                     GATConv(
                        input_size,
                        self.hidden_size // self.num_heads,
                        self.feat_drop,
                        self.attn_drop,
                        self.num_heads,
                        activation=None)
                    for _ in range(edge_type)
                    ]
                    )
                    )
            else:
                self.skips.append(linear_init(self.hidden_size, self.hidden_size, init_type='linear'))
                self.gats.append(
                    nn.LayerList(
                    [GATConv(
                        self.hidden_size,
                        self.hidden_size // self.num_heads,
                        self.feat_drop,
                        self.attn_drop,
                        self.num_heads,
                        activation=None)
                     for _ in range(edge_type)
                    ]
                    )
                    )

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.BatchNorm1D(self.hidden_size,
                                             momentum=0.9, weight_attr=fc_w_attr,
                                            bias_attr=fc_bias_attr, data_format='NC'),
            nn.ReLU(),
            nn.Dropout(p=self.drop),
            nn.Linear(self.hidden_size, self.num_class),
        )

        self.label_mlp = nn.Sequential(
            nn.Linear(2*input_size, self.hidden_size),
            nn.BatchNorm1D(self.hidden_size,
                                             momentum=0.9, weight_attr=fc_w_attr,
                                            bias_attr=fc_bias_attr, data_format='NC'),
            nn.ReLU(),
            nn.Dropout(p=self.drop),
            nn.Linear(self.hidden_size, input_size),
        )

        self.dropout = nn.Dropout(p=self.drop)
        self.input_drop = nn.Dropout(p=0.3)
        
    def get_subgraph_by_masked(self, graph, mask):
        index = L.where(mask)
        if index.shape[0] > 0:
            edges = graph.edges
            sub_edges = paddle.gather(edges, index, axis=0)
            sg = pgl.Graph(sub_edges, num_nodes=graph.num_nodes)
            return sg
        else:
            return None

    def forward(self, graph_list, feature, m2v_feature, label_y, label_idx):
        m2v_fc = self.input_drop(self.m2v_fc(m2v_feature))
        feature = feature + m2v_fc 

        label_embed = self.label_embed(label_y)
        label_embed = self.input_drop(label_embed)
        feature_label = paddle.gather(feature, label_idx)
        label_embed = paddle.concat([label_embed, feature_label], axis=1)
        label_embed = self.label_mlp(label_embed)
        feature = paddle.scatter(feature, label_idx, label_embed, overwrite=True)

        for idx, (sg, sub_index) in enumerate(graph_list):
            temp_feat = []
            skip_feat = paddle.gather(feature, sub_index, axis=0)
            skip_feat = self.skips[idx](skip_feat)
            skip_feat = self.norms[idx][0](skip_feat)
            skip_feat = F.elu(skip_feat)
            temp_feat.append(skip_feat)

            for i in range(self.edge_type):
                masked = sg.edge_feat['edge_type'] == i
                m_sg = self.get_subgraph_by_masked(sg, masked)
                if m_sg is not None:
                    feature_temp = self.gats[idx][i](m_sg, feature)
                    feature_temp = paddle.gather(feature_temp, sub_index, axis=0)
                    feature_temp = self.norms[idx][i+1](feature_temp)
                    feature_temp = F.elu(feature_temp)
                    #skip_feat += feature_temp
                    temp_feat.append(feature_temp)
            temp_feat = paddle.stack(temp_feat, axis=1) # b x 3 x dim
            temp_feat_attn = self.path_attns[idx](temp_feat) # b x 3 x 1
            temp_feat_attn = F.softmax(temp_feat_attn, axis=1)
            temp_feat_attn = paddle.transpose(temp_feat_attn, perm=[0,2,1]) # b x 1 x 3
            skip_feat = paddle.bmm(temp_feat_attn, temp_feat)[:, 0]
            skip_feat = self.path_norms[idx](skip_feat)
            #feature = F.elu(skip_feat)
            feature = self.dropout(skip_feat)
        output = self.mlp(feature)
        return output
