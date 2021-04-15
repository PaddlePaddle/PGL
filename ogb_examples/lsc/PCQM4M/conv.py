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

import pgl
import pgl.nn as gnn
import pgl.nn.functional as GF

from mol_encoder import AtomEncoder, BondEncoder


### GIN convolution along the graph structure
class GINConv(paddle.nn.Layer):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.BatchNorm1D(emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim))

        self.eps = self.create_parameter(
            shape=[1, 1],
            dtype='float32',
            default_initializer=nn.initializer.Constant(value=0))

        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def send_func(self, src_feat, dst_feat, edge_feat):
        return {"h": F.relu(src_feat["x"] + edge_feat["e"])}

    def recv_sum(self, msg):
        return msg.reduce_sum(msg["h"])

    def forward(self, graph, feature, edge_feat):
        edge_embedding = self.bond_encoder(edge_feat)

        msg = graph.send(
            src_feat={"x": feature},
            edge_feat={"e": edge_embedding},
            message_func=self.send_func)

        neigh_feature = graph.recv(msg=msg, reduce_func=self.recv_sum)

        out = (1 + self.eps) * feature + neigh_feature
        out = self.mlp(out)

        return out


### GCN convolution along the graph structure
class GCNConv(paddle.nn.Layer):
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__()

        self.linear = paddle.nn.Linear(emb_dim, emb_dim)
        self.bias = self.create_parameter(shape=[emb_dim], is_bias=True)
        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def send_func(self, src_feat, dst_feat, edge_feat):
        return {"h": F.relu(src_feat["x"] + edge_feat["e"]) * src_feat['norm']}

    def recv_sum(self, msg):
        return msg.reduce_sum(msg["h"])

    def forward(self, graph, feature, edge_feat):
        feature = self.linear(feature)
        edge_embedding = self.bond_encoder(edge_feat)

        norm = GF.degree_norm(graph)

        msg = graph.send(
            src_feat={"x": feature,
                      "norm": norm},
            edge_feat={"e": edge_embedding},
            message_func=self.send_func)

        output = graph.recv(msg=msg, reduce_func=self.recv_sum)

        output = output + self.bias
        output = output * norm

        return output


### GNN to generate node embedding
class GNN_node(paddle.nn.Layer):
    """
    Output:
        node representations
    """

    def __init__(self,
                 num_layers,
                 emb_dim,
                 drop_ratio=0.5,
                 JK="last",
                 residual=False,
                 gnn_type='gin'):
        '''
            emb_dim (int): node embedding dimensionality
            num_layers (int): number of GNN message passing layers
        '''

        super(GNN_node, self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        ###List of GNNs
        self.convs = []
        self.batch_norms = []

        for layer in range(num_layers):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            else:
                ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(paddle.nn.BatchNorm1D(emb_dim))

        self.pool = gnn.GraphPool(pool_type="sum")
        self.convs = nn.LayerList(self.convs)
        self.batch_norms = nn.LayerList(self.batch_norms)

    def forward(self, g):
        x = g.node_feat["feat"]
        edge_feat = g.edge_feat["feat"]

        ### computing input node embedding
        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layers):
            h = self.convs[layer](g, h_list[layer], edge_feat)
            h = self.batch_norms[layer](h)

            if layer == self.num_layers - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(
                    F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layers):
                node_representation += h_list[layer]

        return node_representation


### Virtual GNN to generate node embedding
class GNN_node_Virtualnode(paddle.nn.Layer):
    """
    Output:
        node representations
    """

    def __init__(self,
                 num_layers,
                 emb_dim,
                 drop_ratio=0.5,
                 JK="last",
                 residual=False,
                 gnn_type='gin'):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GNN_node_Virtualnode, self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        ### set the initial virtual node embedding to 0.
        #  self.virtualnode_embedding = paddle.nn.Embedding(1, emb_dim)
        self.virtualnode_embedding = self.create_parameter(
            shape=[1, emb_dim],
            dtype='float32',
            default_initializer=nn.initializer.Constant(value=0.0))

        ### List of GNNs
        self.convs = []
        ### batch norms applied to node embeddings
        self.batch_norms = []

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = []

        for layer in range(num_layers):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            else:
                ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(paddle.nn.BatchNorm1D(emb_dim))

        for layer in range(num_layers - 1):
            self.mlp_virtualnode_list.append(
                nn.Sequential(
                    nn.Linear(emb_dim, emb_dim),
                    nn.BatchNorm1D(emb_dim),
                    nn.ReLU(),
                    nn.Linear(emb_dim, emb_dim),
                    nn.BatchNorm1D(emb_dim), nn.ReLU()))

        self.pool = gnn.GraphPool(pool_type="sum")

        self.convs = nn.LayerList(self.convs)
        self.batch_norms = nn.LayerList(self.batch_norms)
        self.mlp_virtualnode_list = nn.LayerList(self.mlp_virtualnode_list)

    def forward(self, g):
        x = g.node_feat["feat"]
        edge_feat = g.edge_feat["feat"]
        h_list = [self.atom_encoder(x)]

        ### virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding.expand(
            [g.num_graph, self.virtualnode_embedding.shape[-1]])

        for layer in range(self.num_layers):
            ### add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + paddle.gather(
                virtualnode_embedding, g.graph_node_id)

            ### Message passing among graph nodes
            h = self.convs[layer](g, h_list[layer], edge_feat)

            h = self.batch_norms[layer](h)
            if layer == self.num_layers - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(
                    F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            ### update the virtual nodes
            if layer < self.num_layers - 1:
                ### add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = self.pool(
                    g, h_list[layer]) + virtualnode_embedding
                ### transform virtual nodes using MLP

                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(
                        self.mlp_virtualnode_list[layer](
                            virtualnode_embedding_temp),
                        self.drop_ratio,
                        training=self.training)
                else:
                    virtualnode_embedding = F.dropout(
                        self.mlp_virtualnode_list[layer](
                            virtualnode_embedding_temp),
                        self.drop_ratio,
                        training=self.training)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layers):
                node_representation += h_list[layer]

        return node_representation


if __name__ == "__main__":
    pass
