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
import pgl.nn as gnn
from pgl.nn import functional as GF
from mol_encoder import AtomEncoder, BondEncoder


class PNAModel(nn.Layer):
    """
    Implementation of PNA Model
    """

    def __init__(
            self,
            hidden_size,
            out_size,
            aggregators,
            scalers,
            deg_hog,
            pre_layers=1,
            post_layers=1,
            towers=1,
            residual=True,
            batch_norm=True,
            L=3,
            dropout=0.3,
            in_feat_dropout=0.0,
            edge_feat=False,
            num_class=1, ):
        super(PNAModel, self).__init__()
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.aggregators = aggregators
        self.scalers = scalers
        self.pre_layers = pre_layers
        self.post_layers = post_layers
        self.towers = towers
        self.residual = residual
        self.batch_norm = batch_norm
        self.L = L
        self.dropout = dropout
        self.in_feat_dropout = in_feat_dropout
        self.edge_feat = edge_feat
        self.embedding_h = AtomEncoder(emb_dim=hidden_size)
        if self.edge_feat:
            self.embedding_e = BondEncoder(emb_dim=hidden_size)
        self.layers = nn.LayerList()
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.bns = nn.LayerList()
        for i in range(self.L - 1):
            self.layers.append(
                gnn.PNAConv(
                    self.hidden_size,
                    self.hidden_size,
                    self.aggregators,
                    self.scalers,
                    deg_hog,
                    towers=self.towers,
                    pre_layers=self.pre_layers,
                    post_layers=self.post_layers,
                    divide_input=False,
                    use_edge=self.edge_feat))
            if self.batch_norm:
                self.bns.append(nn.BatchNorm1D(hidden_size))
        self.layers.append(
            gnn.PNAConv(
                self.hidden_size,
                self.out_size,
                self.aggregators,
                self.scalers,
                deg_hog,
                towers=self.towers,
                pre_layers=self.pre_layers,
                post_layers=post_layers,
                divide_input=False,
                use_edge=self.edge_feat))
        if self.batch_norm:
            self.bns.append(nn.BatchNorm1D(out_size))
        self.MLP_layer = MLPReadout(out_size, num_class)
        self.pool = gnn.GraphPool("mean")

    def forward(self, graph):
        """
        forward of PNAModel
        """
        h = graph.node_feat['feat']
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        e = None

        if self.edge_feat:
            e = self.embedding_e(graph.edge_feat['feat'])
            e = self.in_feat_dropout(e)
        for i, conv in enumerate(self.layers):
            x = h
            deg = graph.indegree()
            h = conv(graph, h, deg, e)
            if self.batch_norm:
                h = self.bns[i](h)
            h = F.relu(h)  # 

            if self.residual:
                h = h + x
            h = F.dropout(h, self.dropout, training=self.training)

        hg = self.pool(graph, h)
        return self.MLP_layer(hg)


class MLPReadout(nn.Layer):
    """
    An Implementation of MLP layer
    """

    def __init__(self, input_dim, output_dim, L=2):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [
            nn.Linear(
                input_dim // 2**l, input_dim // 2**(l + 1), bias_attr=True)
            for l in range(L)
        ]
        list_FC_layers.append(
            nn.Linear(
                input_dim // 2**L, output_dim, bias_attr=True))
        self.FC_layers = nn.LayerList(list_FC_layers)
        self.L = L

    def forward(self, x):
        """
        forward function of MLPReadout 
        """
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y
