#-*- coding: utf-8 -*-
import os
import sys
import math
sys.path.append("../")
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import pgl.nn as gnn
from pgl.utils.logger import log

import models.conv as CONV
import models.mol_encoder as ME
import models.layers as L
class PretrainBondAngle(paddle.nn.Layer):
    def __init__(self, config):
        super(PretrainBondAngle, self).__init__()
        log.info("Using pretrain bond angle")
        hidden_size = config.emb_dim
        self.bond_angle_pred_linear = nn.Sequential(
            L.Linear(hidden_size*3, hidden_size // 2),
            L.batch_norm_1d(hidden_size // 2),
            nn.Swish(),
            L.Linear(hidden_size//2, hidden_size//4),
            L.batch_norm_1d(hidden_size // 4),
            nn.Swish(),
            L.Linear(hidden_size//4, 1)
        )
        self.loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, node_repr, bond_angle_index, bond_angle, mask):
        node_i, node_j, node_k = bond_angle_index
        node_i_repr = paddle.index_select(node_repr, node_i)
        node_j_repr = paddle.index_select(node_repr, node_j)
        node_k_repr = paddle.index_select(node_repr, node_k)
        
        node_ijk_repr = paddle.concat([node_i_repr, node_j_repr, node_k_repr], axis=1)
        bond_angle_pred = self.bond_angle_pred_linear(node_ijk_repr)
        bond_angle_pred = paddle.masked_select(bond_angle_pred, mask)
        bond_angle_pred = paddle.reshape(bond_angle_pred,[-1,])
        bond_angle = paddle.masked_select(bond_angle,mask)
        bond_angle = paddle.reshape(bond_angle,[-1,])
        loss = self.loss(bond_angle_pred,
                        bond_angle)
        loss = paddle.mean(loss)
        return loss

class PretrainBondLength(paddle.nn.Layer):
    def __init__(self, config):
        super(PretrainBondLength, self).__init__()
        log.info("Using pretrain bond length")
        hidden_size = config.emb_dim
        self.bond_length_pred_linear = nn.Sequential(
            L.Linear(hidden_size*2, hidden_size // 2),
            L.batch_norm_1d(hidden_size // 2),
            nn.Swish(),
            L.Linear(hidden_size//2, hidden_size//4),
            L.batch_norm_1d(hidden_size // 4),
            nn.Swish(),
            L.Linear(hidden_size//4, 1)
        ) 
        self.loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, node_repr, bond_length_index, bond_length, mask):
        node_i, node_j = bond_length_index
        node_i_repr = paddle.index_select(node_repr, node_i)
        node_j_repr = paddle.index_select(node_repr, node_j)
        node_ij_repr = paddle.concat([node_i_repr, node_j_repr], 1)
        bond_length_pred = self.bond_length_pred_linear(node_ij_repr)
        bond_length_pred = paddle.masked_select(bond_length_pred, mask)
        bond_length_pred = paddle.reshape(bond_length_pred,(-1,))
        bond_length = paddle.masked_select(bond_length, mask)
        bond_length = paddle.reshape(bond_length,(-1,))
        loss = self.loss(bond_length_pred,
                         bond_length)
        loss = paddle.mean(loss)
        return loss    
    
class MeanGlobalPool(paddle.nn.Layer):
    def __init__(self, pool_type=None):
        super().__init__()
        self.pool_type = pool_type
        self.pool_fun = gnn.GraphPool("sum")

    def forward(self, graph, nfeat):
        sum_pooled = self.pool_fun(graph, nfeat)
        ones_sum_pooled = self.pool_fun(graph, 
                paddle.ones_like(nfeat, dtype="float32"))
        pooled = sum_pooled / ones_sum_pooled
        return pooled

class BiMapSOPool(paddle.nn.Layer):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.left_linear = paddle.nn.Linear(input_size, hidden_size)
        self.right_linear = paddle.nn.Linear(input_size, hidden_size)
        self.left_norm = L.batch_norm_1d(hidden_size)
        self.right_norm = L.batch_norm_1d(hidden_size)
        self.norm = L.batch_norm_1d(input_size)

    def forward(self, graph, h):
        left_h = self.left_linear(h)
        left_h = self.left_norm(left_h)
        right_h = self.right_linear(h)
        right_h = self.right_norm(right_h)
        h = paddle.matmul(paddle.transpose(left_h, [1,0]), right_h)
        h = paddle.reshape(h, [-1,])
        return h

class GNN(paddle.nn.Layer):
    def __init__(self, config):
        super(GNN, self).__init__()
        log.info("model_type is %s" % self.__class__.__name__)

        self.config = config
        self.pretrain_tasks = config.pretrain_tasks.split(',')
        self.num_layers = config.num_layers
        self.drop_ratio = config.drop_ratio
        self.JK = config.JK
        self.block_num = config.block_num
        self.emb_dim = config.emb_dim
        self.num_tasks = config.num_tasks
        self.residual = config.residual
        self.graph_pooling = config.graph_pooling

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        self.gnn_blocks = paddle.nn.LayerList()
        for i in range(self.config.block_num):
            self.gnn_blocks.append(getattr(CONV, self.config.gnn_type)(config))

        hidden_size = self.emb_dim * self.block_num
        ### Pooling function to generate whole-graph embeddings
        if self.config.graph_pooling == "bisop":
            pass
        else:
            self.pool = MeanGlobalPool()

        if self.config.clf_layers == 3:
            log.info("clf_layers is 3")
            self.graph_pred_linear = nn.Sequential(
                    L.Linear(hidden_size, hidden_size // 2),
                    L.batch_norm_1d(hidden_size // 2),
                    nn.Swish(),
                    L.Linear(hidden_size // 2, hidden_size // 4),
                    L.batch_norm_1d(hidden_size // 4),
                    nn.Swish(),
                    L.Linear(hidden_size // 4, self.num_tasks)
                    )
        elif self.config.clf_layers == 2:
            log.info("clf_layers is 2")
            self.graph_pred_linear = nn.Sequential(
                L.Linear(hidden_size, hidden_size // 2),
                L.batch_norm_1d(hidden_size // 2),
                nn.Swish(),
                L.Linear(hidden_size // 2, self.num_tasks)
            )
        else:
            self.graph_pred_linear = L.Linear(hidden_size, self.num_tasks)
            
        if 'Con' in self.pretrain_tasks:
            self.context_loss = nn.CrossEntropyLoss()
            self.contextmlp = nn.Sequential(
                                L.Linear(self.emb_dim, self.emb_dim//2),
                                L.batch_norm_1d(self.emb_dim//2),
                                nn.Swish(),
                                L.Linear(self.emb_dim//2, 5000)
                                )
        if 'Ba' in self.pretrain_tasks:
            self.pretrain_bond_angle = PretrainBondAngle(config)
        if 'Bl' in self.pretrain_tasks:
            self.pretrain_bond_length = PretrainBondLength(config)

    def forward(self, feed_dict, return_graph=False, return_pretrain=True):
        h_nodes = []
        # adding pretrain relevant feature information
        bond_angle_index = feed_dict["bond_angle_index"]
        bond_angle = feed_dict["bond_angle"]
        bond_angle_mask = feed_dict["bond_angle_mask"]
        edge_index = feed_dict["edge_index"]
        edge_attr_float = feed_dict["edge_attr_float"]
        edge_attr_float_mask = feed_dict["edge_attr_float_mask"]
        tid = feed_dict["tid"]
        
        for i in range(self.config.block_num):
            h_nodes.append(self.gnn_blocks[i](feed_dict))
        h_node = paddle.concat(h_nodes, axis=-1)

        pretrain_losses = {}
        if 'Con' in self.pretrain_tasks:
            pred2 = self.contextmlp(h_node)
#             print(f"pred2 shape :{pred2.shape}, tid shape: {tid.shape}")
            context_loss = self.context_loss(pred2, tid)
            pretrain_losses['Con'] = context_loss
        if 'Ba' in self.pretrain_tasks:
            pretrain_losses['Ba'] = self.pretrain_bond_angle(
                    h_node, bond_angle_index, bond_angle, bond_angle_mask)
        if 'Bl' in self.pretrain_tasks:
            pretrain_losses['Bl'] = self.pretrain_bond_length(
                    h_node, edge_index, edge_attr_float, edge_attr_float_mask)
            
        if self.graph_pooling == "bisop":
            h_graph = self.pool(h_node)
        else:
            h_graph = self.pool(feed_dict['graph'], h_node)
        output = self.graph_pred_linear(h_graph)

        #  if self.training:
        #      return output
        #  else:
        #      # At inference time, relu is applied to output to ensure positivity
        #      return paddle.clip(output, min=0, max=50)
        #return output
        if return_graph and return_pretrain:
            return output, pretrain_losses
        if return_pretrain:
            return pretrain_losses
        if return_graph:
            return output

class MLP(paddle.nn.Layer):
    def __init__(self, config):
        super(MLP, self).__init__()
        log.info("model_type is %s" % self.__class__.__name__)

        self.config = config
        self.num_layers = config.num_layers
        self.drop_ratio = config.drop_ratio
        self.JK = config.JK
        self.emb_dim = config.emb_dim
        self.num_tasks = config.num_tasks
        self.residual = config.residual
        self.graph_pooling = config.graph_pooling

        self.atom_encoder = getattr(ME, self.config.atom_enc_type, ME.AtomEncoder)(
                emb_dim=self.emb_dim)

        self.mlp = L.MLP([self.emb_dim, 2 * self.emb_dim, self.emb_dim], 
                       norm=self.config.norm,
                       last_lin=True)

        ### Pooling function to generate whole-graph embeddings
        self.pool = MeanGlobalPool()
        self.graph_pred_linear = L.Linear(self.emb_dim, self.num_tasks)

    def forward(self, feed_dict):
        g = feed_dict['graph']
        nfeat = g.node_feat['feat']
        h = self.atom_encoder(nfeat)
        h = self.mlp(h)
        h_graph = self.pool(g, h)
        output = self.graph_pred_linear(h_graph)
        return output

if __name__ == '__main__':
    GNN(num_tasks = 10)
