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

import pgl
import paddle.nn as nn
import paddle.nn.functional as F


class GCN(nn.Layer):
    """Implement of GCN
    """

    def __init__(self,
                 input_size,
                 num_class,
                 num_layers=1,
                 hidden_size=64,
                 dropout=0.5,
                 **kwargs):
        super(GCN, self).__init__()
        self.num_class = num_class
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.gcns = nn.LayerList()
        for i in range(self.num_layers):
            if i == 0:
                self.gcns.append(
                    pgl.nn.GCNConv(
                        input_size,
                        self.hidden_size,
                        activation="relu",
                        norm=True))
            else:
                self.gcns.append(
                    pgl.nn.GCNConv(
                        self.hidden_size,
                        self.hidden_size,
                        activation="relu",
                        norm=True))
            self.gcns.append(nn.Dropout(self.dropout))
        self.gcns.append(pgl.nn.GCNConv(self.hidden_size, self.num_class))

    def forward(self, graph, feature):
        for m in self.gcns:
            if isinstance(m, nn.Dropout):
                feature = m(feature)
            else:
                feature = m(graph, feature)
        return feature


class GAT(nn.Layer):
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
                 **kwargs):
        super(GAT, self).__init__()
        self.num_class = num_class
        self.num_layers = num_layers
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.gats = nn.LayerList()
        for i in range(self.num_layers):
            if i == 0:
                self.gats.append(
                    pgl.nn.GATConv(
                        input_size,
                        self.hidden_size,
                        self.feat_drop,
                        self.attn_drop,
                        self.num_heads,
                        activation='elu'))
            elif i == (self.num_layers - 1):
                self.gats.append(
                    pgl.nn.GATConv(
                        self.num_heads * self.hidden_size,
                        self.num_class,
                        self.feat_drop,
                        self.attn_drop,
                        1,
                        concat=False,
                        activation=None))
            else:
                self.gats.append(
                    pgl.nn.GATConv(
                        self.num_heads * self.hidden_size,
                        self.hidden_size,
                        self.feat_drop,
                        self.attn_drop,
                        self.num_heads,
                        activation='elu'))

    def forward(self, graph, feature):
        for m in self.gats:
            feature = m(graph, feature)
        return feature


class Transformer(nn.Layer):
    """Implement of TransformerConv
    """

    def __init__(self,
                 input_size,
                 num_class,
                 num_layers=1,
                 feat_drop=0.6,
                 attn_drop=0.6,
                 num_heads=8,
                 hidden_size=8,
                 **kwargs):
        super(Transformer, self).__init__()
        self.num_class = num_class
        self.num_layers = num_layers
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.trans = nn.LayerList()

        for i in range(self.num_layers):
            if i == 0:
                self.trans.append(
                    pgl.nn.TransformerConv(
                        input_size,
                        self.hidden_size,
                        self.num_heads,
                        self.feat_drop,
                        self.attn_drop,
                        skip_feat=False,
                        activation='relu'))

            elif i == (self.num_layers - 1):
                self.trans.append(
                    pgl.nn.TransformerConv(
                        self.num_heads * self.hidden_size,
                        self.num_class,
                        self.num_heads,
                        self.feat_drop,
                        self.attn_drop,
                        concat=False,
                        skip_feat=False,
                        layer_norm=False,
                        activation=None))
            else:
                self.trans.append(
                    pgl.nn.TransformerConv(
                        self.num_head * self.hidden_size,
                        self.hidden_size,
                        self.num_heads,
                        self.feat_drop,
                        self.attn_drop,
                        skip_feat=False,
                        activation='relu'))

    def forward(self, graph, feature):
        for m in self.trans:
            feature = m(graph, feature)
        return feature


class APPNP(nn.Layer):
    """Implement of APPNP 
    """

    def __init__(self,
                 input_size,
                 num_class,
                 num_layers=1,
                 hidden_size=64,
                 dropout=0.5,
                 k_hop=10,
                 alpha=0.1,
                 **kwargs):
        super(APPNP, self).__init__()
        self.num_class = num_class
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.alpha = alpha
        self.k_hop = k_hop

        self.mlps = nn.LayerList()
        self.mlps.append(nn.Linear(input_size, self.hidden_size))
        self.drop_fn = nn.Dropout(self.dropout)
        for _ in range(self.num_layers - 1):
            self.mlps.append(nn.Linear(self.hidden_size, self.hidden_size))

        self.output = nn.Linear(self.hidden_size, num_class)
        self.appnp = pgl.nn.APPNP(alpha=self.alpha, k_hop=self.k_hop)

    def forward(self, graph, feature):
        for m in self.mlps:
            feature = self.drop_fn(feature)
            feature = m(feature)
            feature = F.relu(feature)
        feature = self.drop_fn(feature)
        feature = self.output(feature)
        feature = self.appnp(graph, feature)
        return feature


class SGC(nn.Layer):
    """Implement of SGC
    """

    def __init__(self, input_size, num_class, num_layers=1, **kwargs):
        super(SGC, self).__init__()
        self.num_class = num_class
        self.num_layers = num_layers
        self.sgc_layer = pgl.nn.SGCConv(input_size=input_size, output_size=num_class, k_hop=num_layers)

    def forward(self, graph, feature):
        feature = graph.node_feat["words"]
        feature = self.sgc_layer(graph, feature)
        return feature


class GCNII(nn.Layer):
    """Implement of GCNII
    """

    def __init__(self,
                 input_size,
                 num_class,
                 num_layers=1,
                 hidden_size=64,
                 dropout=0.6,
                 lambda_l=0.5,
                 alpha=0.1,
                 k_hop=64,
                 **kwargs):
        super(GCNII, self).__init__()
        self.num_class = num_class
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.lambda_l = lambda_l
        self.alpha = alpha
        self.k_hop = k_hop 

        self.mlps = nn.LayerList()
        self.mlps.append(nn.Linear(input_size, self.hidden_size))
        self.drop_fn = nn.Dropout(self.dropout)
        for _ in range(self.num_layers - 1):
            self.mlps.append(nn.Linear(self.hidden_size, self.hidden_size))

        self.output = nn.Linear(self.hidden_size, num_class)
        self.gcnii = pgl.nn.GCNII(
            hidden_size=self.hidden_size,
            activation="relu",
            lambda_l=self.lambda_l,
            alpha=self.alpha,
            k_hop=self.k_hop,
            dropout=self.dropout)

    def forward(self, graph, feature):
        for m in self.mlps:
            feature = m(feature)
            feature = F.relu(feature)
            feature = self.drop_fn(feature)
        feature = self.gcnii(graph, feature)
        feature = self.output(feature)
        return feature
