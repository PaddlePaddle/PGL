import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import pgl
import pgl.nn as gnn
from pgl.utils.logger import log

import models.mol_encoder as ME
import models.layers as L

class LiteGEM(paddle.nn.Layer):
    def __init__(self, config, with_efeat=False):
        super(LiteGEM, self).__init__()
        log.info("gnn_type is %s" % self.__class__.__name__)

        self.config = config
        self.with_efeat = with_efeat
        self.num_layers = config.num_layers
        self.drop_ratio = config.drop_ratio
        self.virtual_node = config.virtual_node
        self.emb_dim = config.emb_dim
        self.norm = config.norm

        self.gnns = paddle.nn.LayerList()
        self.norms = paddle.nn.LayerList()

        if self.virtual_node:
            log.info("using virtual node in %s" % self.__class__.__name__)
            self.mlp_virtualnode_list = paddle.nn.LayerList()

            self.virtualnode_embedding = self.create_parameter(
                shape=[1, self.emb_dim],
                dtype='float32',
                default_initializer=nn.initializer.Constant(value=0.0))

            for layer in range(self.num_layers - 1):
                self.mlp_virtualnode_list.append(L.MLP([self.emb_dim] * 3,
                                                       norm=self.norm))

        for layer in range(self.num_layers):
            self.gnns.append(L.LiteGEMConv(config, with_efeat=not self.with_efeat))
            self.norms.append(L.norm_layer(self.norm, self.emb_dim))

        self.atom_encoder = getattr(ME, self.config.atom_enc_type, ME.AtomEncoder)(
                emb_dim=self.emb_dim)
        if self.config.exfeat:
            self.atom_encoder_float = ME.AtomEncoderFloat(emb_dim=self.emb_dim)

        if self.with_efeat:
            self.bond_encoder = getattr(ME, self.config.bond_enc_type, ME.BondEncoder)(
                    emb_dim=self.emb_dim)

        self.pool = gnn.GraphPool(pool_type="sum")

        if self.config.appnp_k is not None:
            self.appnp = gnn.APPNP(k_hop=self.config.appnp_k, alpha=self.config.appnp_a)

        if self.config.graphnorm is not None:
            self.gn = gnn.GraphNorm()

    def forward(self, feed_dict):
        g = feed_dict["graph"]
        x = g.node_feat["feat"]
        edge_feat = g.edge_feat["feat"]

        h = self.atom_encoder(x)
        if self.config.exfeat:
            h += self.atom_encoder_float(g.node_feat["feat_float"])
        #  print("atom_encoder: ", np.sum(h.numpy()))

        if self.virtual_node:
            virtualnode_embedding = self.virtualnode_embedding.expand(
                    [g.num_graph, self.virtualnode_embedding.shape[-1]])
            h = h + paddle.gather(virtualnode_embedding, g.graph_node_id)
            #  print("virt0: ", np.sum(h.numpy()))

        if self.with_efeat:
            edge_emb = self.bond_encoder(edge_feat)
        else:
            edge_emb = edge_feat

        h = self.gnns[0](g, h, edge_emb)
        if self.config.graphnorm:
            h = self.gn(g, h)

        #  print("h0: ", np.sum(h.numpy()))
        for layer in range(1, self.num_layers):
            h1 = self.norms[layer-1](h)
            h2 = F.swish(h1)
            h2 = F.dropout(h2, p=self.drop_ratio, training=self.training)

            if self.virtual_node:
                virtualnode_embedding_temp = self.pool(g, h2) + virtualnode_embedding
                virtualnode_embedding = self.mlp_virtualnode_list[layer-1](virtualnode_embedding_temp)
                virtualnode_embedding  = F.dropout(
                        virtualnode_embedding,
                        self.drop_ratio,
                        training=self.training)

                h2 = h2 + paddle.gather(virtualnode_embedding, g.graph_node_id)
                #  print("virt_h%s: " % (layer), np.sum(h2.numpy()))

            h = self.gnns[layer](g, h2, edge_emb) + h
            if self.config.graphnorm:
                h = self.gn(g, h)
            #  print("h%s: " % (layer), np.sum(h.numpy()))

        h = self.norms[self.num_layers-1](h)
        h = F.dropout(h, p=self.drop_ratio, training=self.training)

        if self.config.appnp_k is not None:
            h = self.appnp(g, h)
        #  print("node_repr: ", np.sum(h.numpy()))
        node_representation = h
        return node_representation

class GNNVirt(paddle.nn.Layer):
    def __init__(self, config):
        super(GNNVirt, self).__init__()
        log.info("gnn_type is %s" % self.__class__.__name__)
        self.config = config

        self.atom_encoder = getattr(ME, self.config.atom_enc_type, ME.AtomEncoder)(
                self.config.emb_dim)

        self.virtualnode_embedding = self.create_parameter(
            shape=[1, self.config.emb_dim],
            dtype='float32',
            default_initializer=nn.initializer.Constant(value=0.0))

        self.convs = paddle.nn.LayerList()
        self.batch_norms = paddle.nn.LayerList()
        self.mlp_virtualnode_list = paddle.nn.LayerList()

        for layer in range(self.config.num_layers):
            self.convs.append(getattr(L, self.config.layer_type)(self.config))
            self.batch_norms.append(L.batch_norm_1d(self.config.emb_dim))

        for layer in range(self.config.num_layers - 1):
            self.mlp_virtualnode_list.append(
                    nn.Sequential(L.Linear(self.config.emb_dim, self.config.emb_dim), 
                        L.batch_norm_1d(self.config.emb_dim), 
                        nn.Swish(),
                        L.Linear(self.config.emb_dim, self.config.emb_dim), 
                        L.batch_norm_1d(self.config.emb_dim), 
                        nn.Swish())
                    )

        self.pool = gnn.GraphPool(pool_type="sum")

    def forward(self, feed_dict):
        g = feed_dict["graph"]
        x = g.node_feat["feat"]
        edge_feat = g.edge_feat["feat"]
        
        h_list = [self.atom_encoder(x)]

        virtualnode_embedding = self.virtualnode_embedding.expand(
                [g.num_graph, self.virtualnode_embedding.shape[-1]])

        for layer in range(self.config.num_layers):
            h_list[layer] = h_list[layer] + \
                    paddle.gather(virtualnode_embedding, g.graph_node_id)

            ### Message passing among graph nodes
            h = self.convs[layer](g, h_list[layer], edge_feat)
            h = self.batch_norms[layer](h)
            if layer == self.config.num_layers - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.config.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.swish(h), self.config.drop_ratio, training = self.training)

            if self.config.residual:
                h = h + h_list[layer]

            h_list.append(h)

            ### update the virtual nodes
            if layer < self.config.num_layers - 1:
                ### add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = self.pool(g, h_list[layer]) + virtualnode_embedding
                ### transform virtual nodes using MLP

                if self.config.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp),
                        self.config.drop_ratio, training = self.training)
                else:
                    virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp),
                        self.config.drop_ratio, training = self.training)

        ### Different implementations of Jk-concat
        if self.config.JK == "last":
            node_representation = h_list[-1]
        elif self.config.JK == "sum":
            node_representation = 0
            for layer in range(self.config.num_layers):
                node_representation += h_list[layer]
        
        return node_representation



### Virtual GNN to generate node embedding
class JuncGNNVirt(paddle.nn.Layer):
    """
    Output:
        node representations
    """
    def __init__(self, config):
        super(JuncGNNVirt, self).__init__()
        log.info("gnn_type is %s" % self.__class__.__name__)
        self.config = config
        self.num_layers = config.num_layers
        self.drop_ratio = config.drop_ratio
        self.JK = config.JK
        self.residual = config.residual
        self.emb_dim = config.emb_dim
        self.gnn_type = config.gnn_type
        self.layer_type = config.layer_type

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = getattr(ME, self.config.atom_enc_type, ME.AtomEncoder)(
                self.emb_dim)

        self.junc_embed = paddle.nn.Embedding(6000, self.emb_dim)

        ### set the initial virtual node embedding to 0.
        #  self.virtualnode_embedding = paddle.nn.Embedding(1, emb_dim)
        #  torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
        self.virtualnode_embedding = self.create_parameter(
            shape=[1, self.emb_dim],
            dtype='float32',
            default_initializer=nn.initializer.Constant(value=0.0))

        ### List of GNNs
        self.convs = nn.LayerList()
        ### batch norms applied to node embeddings
        self.batch_norms = nn.LayerList()

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = nn.LayerList()

        self.junc_convs = nn.LayerList()

        for layer in range(self.num_layers):
            self.convs.append(getattr(L, self.layer_type)(self.config))
            self.junc_convs.append(gnn.GINConv(self.emb_dim, self.emb_dim))

            self.batch_norms.append(L.batch_norm_1d(self.emb_dim))

        for layer in range(self.num_layers - 1):
            self.mlp_virtualnode_list.append(
                    nn.Sequential(L.Linear(self.emb_dim, self.emb_dim), 
                        L.batch_norm_1d(self.emb_dim), 
                        nn.Swish(),
                        L.Linear(self.emb_dim, self.emb_dim), 
                        L.batch_norm_1d(self.emb_dim), 
                        nn.Swish())
                    )

        self.pool = gnn.GraphPool(pool_type="sum")


    def forward(self, feed_dict):
        g = feed_dict['graph']

        x = g.node_feat["feat"]
        edge_feat = g.edge_feat["feat"]
        h_list = [self.atom_encoder(x)]

        ### virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding.expand(
                [g.num_graph, self.virtualnode_embedding.shape[-1]])

        junc_feat = self.junc_embed(feed_dict['junc_graph'].node_feat['feat'])
        junc_feat = paddle.squeeze(junc_feat, axis=1)
        for layer in range(self.num_layers):
            ### add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + paddle.gather(virtualnode_embedding, g.graph_node_id)

            ### Message passing among graph nodes
            h = self.convs[layer](g, h_list[layer], edge_feat)

            h = self.batch_norms[layer](h)
            if layer == self.num_layers - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.swish(h), self.drop_ratio, training = self.training)

            if self.residual:
                h = h + h_list[layer]

            # junction tree aggr
            atom_index = feed_dict['mol2junc'][:, 0]
            junc_index = feed_dict['mol2junc'][:, 1]
            gather_h = paddle.gather(h, atom_index)
            out_dim = gather_h.shape[-1]
            num = feed_dict['junc_graph'].num_nodes
            init_h = paddle.zeros(shape=[num, out_dim], dtype=gather_h.dtype)
            junc_h = paddle.scatter(init_h, junc_index, gather_h, overwrite=False)
            # node feature of junction tree
            junc_h = junc_feat + junc_h

            junc_h = self.junc_convs[layer](feed_dict['junc_graph'], junc_h)

            junc_h = paddle.gather(junc_h, junc_index)
            init_h = paddle.zeros(shape=[feed_dict['graph'].num_nodes, out_dim], dtype=h.dtype)
            sct_h = paddle.scatter(init_h, atom_index, junc_h, overwrite=False)
            h = h + sct_h

            h_list.append(h)

            ### update the virtual nodes
            if layer < self.num_layers - 1:
                ### add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = self.pool(g, h_list[layer]) + virtualnode_embedding
                ### transform virtual nodes using MLP

                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)
                else:
                    virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)

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
