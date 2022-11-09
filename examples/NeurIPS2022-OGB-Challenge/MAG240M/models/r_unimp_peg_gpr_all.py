# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from .gat_conv_peg import linear_init, GATConv
import numpy as np

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
                activation=None,
                alpha=None,
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
        self.path_norms = nn.LayerList()
        self.label_embed = nn.Embedding(num_class, input_size)
        self.gpr_attn = None
        if alpha:
            k_hop = num_layers
            self.gpr_attn = alpha * (1 - alpha) ** np.arange(k_hop + 1)
            self.gpr_attn[-1] = (1 - alpha) ** k_hop
        # whether to use learnable year_pos
        if 'm2v_dim' in kwargs:
            self.m2v_dim = kwargs['m2v_dim']
        else:
            self.m2v_dim = 64
        # self.m2v_fc = linear_init(self.m2v_dim, input_size, init_type='linear')
        self.p2p_fc = linear_init(self.m2v_dim, input_size, init_type='linear')
        fc_w_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(1.0))
        fc_bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(0.0))
        for i in range(self.num_layers):
            self.path_attns.append(linear_init(self.hidden_size, 1, init_type='linear'))
            self.path_norms.append(nn.BatchNorm1D(self.hidden_size,
                                                  momentum=0.9, weight_attr=fc_w_attr,
                                                  bias_attr=fc_bias_attr, data_format='NC'))
            self.norms.append(nn.LayerList([nn.BatchNorm1D(self.hidden_size,
                                             momentum=0.9, weight_attr=fc_w_attr,
                                            bias_attr=fc_bias_attr, data_format='NC') 
                                                for _ in range(edge_type+1)]))
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

    def get_subgraph_by_edge_split(self, idx, edge_src, edge_dst, edge_split, num_nodes):
        start = 0 if idx == 0 else edge_split[idx - 1]
        if start == edge_split[idx]:
            return None
        new_edge_src = edge_src[start: edge_split[idx]].reshape([-1, 1])
        new_edge_dst = edge_dst[start: edge_split[idx]].reshape([-1, 1])
        graph = pgl.Graph(num_nodes=num_nodes,
                          edges=paddle.concat([new_edge_src, new_edge_dst], axis=1))
        return graph


    def forward(self, graph_list, feature, m2v_feature, p2p_feature, label_y, label_idx, batch_nodes, pos):
            
        # whether to use fuse input
        feature = feature + pos
        p2p_fc = self.input_drop(self.p2p_fc(p2p_feature))
        # m2v_fc = self.input_drop(self.m2v_fc(m2v_feature))
        feature = feature + p2p_fc
        
        label_embed = self.label_embed(label_y)
        label_embed = self.input_drop(label_embed)
        feature_label = paddle.gather(feature, label_idx)
        label_embed = paddle.concat([label_embed, feature_label], axis=1)
        label_embed = self.label_mlp(label_embed)
        feature = paddle.scatter(feature, label_idx, label_embed, overwrite=True)
        pp = pos.astype("float32") + p2p_fc
		
        for idx, (edge_src, edge_dst, edge_split, next_num_nodes, num_nodes) in enumerate(graph_list):
            temp_feat = []
            # post-smoothing
            skip_feat = feature[:next_num_nodes]  # 当前采样图中心点对应的特征
            skip_feat = self.skips[idx](skip_feat)
            skip_feat = self.norms[idx][0](skip_feat)
            skip_feat = F.elu(skip_feat)
            temp_feat.append(skip_feat)
            if self.gpr_attn is not None:
                if idx == 0:
                    gpr_feature = self.gpr_attn[idx] * skip_feat
                else:
                    gpr_feature = gpr_feature[:next_num_nodes] + self.gpr_attn[idx] * skip_feat
            for i in range(self.edge_type):
                m_sg = self.get_subgraph_by_edge_split(i, edge_src, edge_dst, edge_split, num_nodes)
                if m_sg is not None:
                    # 分别gat, 并且取采样图中心对应特征
                    feature_temp = self.gats[idx][i](m_sg, feature, pp)
                    feature_temp = self.norms[idx][i + 1](feature_temp[:next_num_nodes])
                    feature_temp = F.elu(feature_temp)
                    temp_feat.append(feature_temp)
            # all_type_feaute fuse using att
            temp_feat = paddle.stack(temp_feat, axis=1)
            temp_feat_attn = self.path_attns[idx](temp_feat)
            temp_feat_attn = F.softmax(temp_feat_attn, axis=1)
            temp_feat_attn = paddle.transpose(temp_feat_attn, perm=[0, 2, 1])
            skip_feat = paddle.bmm(temp_feat_attn, temp_feat)[:, 0]
            skip_feat = self.path_norms[idx](skip_feat)
            feature = self.dropout(skip_feat)
        gpr_feature = (gpr_feature[:next_num_nodes] + self.gpr_attn[-1] * feature ) if self.gpr_attn is not None else feature
        output = self.mlp(gpr_feature)
        return output
