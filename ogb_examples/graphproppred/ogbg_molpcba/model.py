# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
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


class MLP(nn.Layer):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.main = [
            nn.Linear(in_dim, 2 * in_dim), nn.BatchNorm1D(2 * in_dim),
            nn.ReLU()
        ]
        self.main.append(nn.Linear(2 * in_dim, out_dim))
        self.main = nn.Sequential(*self.main)

    def forward(self, x):
        return self.main(x)


#################
# Embeddings
#################
class OGBMolEmbedding(nn.Layer):
    def __init__(self, dim, embed_edge=True, x_as_list=False):
        super(OGBMolEmbedding, self).__init__()
        self.atom_embedding = AtomEncoder(emb_dim=dim)
        if embed_edge:
            self.edge_embedding = BondEncoder(emb_dim=dim)
        self.x_as_list = x_as_list

    def forward(self, graph):
        graph.node_feat["atom_feat"] = self.atom_embedding(graph.node_feat[
            "feat"])
        if self.x_as_list:
            graph.node_feat["atom_feat"] = [graph.node_feat["atom_feat"]]
        if hasattr(self, 'edge_embedding'):
            graph.edge_feat["bond_feat"] = self.edge_embedding(graph.edge_feat[
                "feat"])
        return graph


class VNAgg(nn.Layer):
    def __init__(self, dim, conv_type="gin"):
        super(VNAgg, self).__init__()
        self.conv_type = conv_type
        if "gin" in conv_type:
            self.mlp = nn.Sequential(
                MLP(dim, dim), nn.BatchNorm1D(dim), nn.ReLU())
        elif "gcn" in conv_type:
            self.W0 = nn.Linear(dim, dim)
            self.W1 = nn.Linear(dim, dim)
            self.nl_bn = nn.Sequential(nn.BatchNorm1D(dim), nn.ReLU())
        else:
            raise NotImplementedError('Unrecognised model conv : {}'.format(
                conv_type))

    def forward(self, virtual_node, embeddings, graph_node_id):
        if graph_node_id.shape[
                0] > 0:  # ...or the operation will crash for empty graphs
            G = pgl.math.segment_sum(embeddings, graph_node_id)
        else:
            G = paddle.zeros_like(virtual_node)

        if "gin" in self.conv_type:
            virtual_node = virtual_node + G
            virtual_node = self.mlp(virtual_node)

        elif "gcn" in self.conv_type:
            virtual_node = self.W0(virtual_node) + self.W1(G)
            virtual_node = self.nl_bn(virtual_node)
        else:
            raise NotImplementedError('Unrecognised model conv : {}'.format(
                self.conv_type))
        return virtual_node


class ConvBlock(nn.Layer):
    def __init__(self,
                 dim,
                 dropout=0.5,
                 activation=F.relu,
                 virtual_node=False,
                 virtual_node_agg=True,
                 k=4,
                 last_layer=False,
                 conv_type='gin',
                 edge_embedding=None):
        super().__init__()
        self.edge_embed = edge_embedding
        self.conv_type = conv_type
        if conv_type == 'gin+':
            self.conv = GINEPLUS(MLP(dim, dim), dim, k=k)
        elif conv_type == 'gcn':
            self.conv = gnn.GCNConv(dim, dim)
        self.norm = nn.BatchNorm1D(dim)
        self.act = activation or None
        self.last_layer = last_layer

        self.dropout_ratio = dropout

        self.virtual_node = virtual_node
        self.virtual_node_agg = virtual_node_agg
        if self.virtual_node and self.virtual_node_agg:
            self.vn_aggregator = VNAgg(dim, conv_type=conv_type)

    def forward(self, graph):
        ea = graph.edge_feat["feat"]

        multi_hop_graph = graph.multi_hop_graphs

        h = graph.node_feat["atom_feat"]
        if self.virtual_node:
            if self.conv_type == 'gin+':
                h[0] = h[0] + paddle.gather(graph.virtual_node,
                                            graph.graph_node_id)
            else:
                h = h + paddle.gather(virtual_node, graph.graph_node_id)

        if self.conv_type == 'gcn':
            H = self.conv(graph, h)
        elif self.conv_type == "gin+":
            H = self.conv(multi_hop_graph, h, self.edge_embed(ea))

        if self.conv_type == 'gin+':
            h = H[0]
        else:
            h = H
        h = self.norm(h)
        if not self.last_layer:
            h = self.act(h)
        h = F.dropout(h, self.dropout_ratio, training=self.training)

        if self.virtual_node and self.virtual_node_agg:
            v = self.vn_aggregator(graph.virtual_node, h, graph.graph_node_id)
            v = F.dropout(v, self.dropout_ratio, training=self.training)
            graph.virtual_node = v

        if self.conv_type == 'gin+':
            H[0] = h
            h = H
        graph.node_feat["atom_feat"] = h
        return graph


class GlobalPool(nn.Layer):
    def __init__(self, fun):
        super().__init__()
        self.pool_type = fun
        self.pool_fun = gnn.GraphPool()

    def forward(self, graph):
        sum_pooled = self.pool_fun(
            graph, graph.node_feat["atom_feat"], pool_type="SUM")
        ones_sum_pooled = self.pool_fun(
            graph,
            paddle.ones_like(
                graph.node_feat["atom_feat"], dtype="float32"),
            pool_type="SUM")
        pooled = sum_pooled / ones_sum_pooled
        return pooled


class ClassifierNetwork(nn.Layer):
    def __init__(self,
                 hidden=100,
                 out_dim=128,
                 layers=3,
                 dropout=0.5,
                 virtual_node=False,
                 k=4,
                 conv_type='gin',
                 appnp_hop=5,
                 alpha=0.2):
        super(ClassifierNetwork, self).__init__()
        self.k = k
        self.conv_type = conv_type
        convs = [
            ConvBlock(
                hidden,
                dropout=dropout,
                virtual_node=virtual_node,
                k=min(i + 1, k),
                conv_type=conv_type,
                edge_embedding=BondEncoder(emb_dim=hidden))
            for i in range(layers - 1)
        ]
        convs.append(
            ConvBlock(
                hidden,
                dropout=dropout,
                virtual_node=virtual_node,
                virtual_node_agg=False,  # on last layer, use but do not update virtual node
                last_layer=True,
                k=min(layers, k),
                conv_type=conv_type,
                edge_embedding=BondEncoder(emb_dim=hidden)))

        self.main = nn.Sequential(
            OGBMolEmbedding(
                hidden, embed_edge=False, x_as_list=(conv_type == 'gin+')),
            *convs)

        self.aggregate = nn.Sequential(
            GlobalPool('mean'), nn.Linear(hidden, out_dim))

        self.virtual_node = virtual_node
        if self.virtual_node:
            self.v0 = self.create_parameter(
                shape=[1, hidden],
                dtype="float32",
                default_initializer=nn.initializer.Constant(value=0.0))

        self.appnp = gnn.APPNP(alpha=alpha, k_hop=appnp_hop, self_loop=True)

    def forward(self, graph):
        if self.virtual_node:
            graph.virtual_node = self.v0.expand(
                [graph.num_graph, self.v0.shape[-1]])
        g = self.main(graph)
        if self.conv_type == 'gin+':
            g.node_feat["atom_feat"] = g.node_feat["atom_feat"][0]

        g.node_feat['atom_feat'] = self.appnp(g, g.node_feat['atom_feat'])

        return self.aggregate(g)

    def degree_norm(self, g):
        degree = g.indegree() + 1  # self loop
        norm = paddle.cast(degree, dtype=paddle.get_default_dtype())
        norm = paddle.clip(norm, min=1.0)
        norm = paddle.pow(norm, -0.5)
        norm = paddle.reshape(norm, [-1, 1])
        return norm


class GINEPLUS(nn.Layer):
    def __init__(self, fun, dim, k=4, **kwargs):
        super(GINEPLUS, self).__init__()
        self.k = k
        self.nn = fun
        self.eps = self.create_parameter(
            shape=[k + 1, dim],
            dtype="float32",
            default_initializer=nn.initializer.Constant(value=0.0))

    def send_add_edge(self, src_feat, dst_feat, edge_feat):
        if "e" in edge_feat:
            return {"h": F.relu(src_feat["x"] + edge_feat["e"])}
        else:
            return {"h": F.relu(src_feat["x"])}

    def recv_sum(self, msg):
        return msg.reduce_sum(msg["h"])

    def forward(self, multi_hop_graphs, XX, edge_attr):
        """Warning, XX is now a list of previous xs, with x[0] being the last layer"""
        result = (1 + self.eps[0]) * XX[0]
        for i, x in enumerate(XX):
            if i >= self.k:
                break
            if i == 0:
                msg = multi_hop_graphs[i + 1].send(
                    src_feat={"x": x},
                    edge_feat={"e": edge_attr},
                    message_func=self.send_add_edge)
            else:
                msg = multi_hop_graphs[i + 1].send(
                    src_feat={"x": x}, message_func=self.send_add_edge)
            out = multi_hop_graphs[i + 1].recv(
                msg=msg, reduce_func=self.recv_sum)
            result += (1 + self.eps[i + 1]) * out
        result = self.nn(result)
        return [result] + XX


if __name__ == "__main__":
    # check
    model = ClassifierNetwork(conv_type="gcn")
    dataset = PglGraphPropPredDataset(name="ogbg-molpcba")
    graph, label = dataset[0]
    out = model(graph.tensor(inplace=False))
    print(out)
