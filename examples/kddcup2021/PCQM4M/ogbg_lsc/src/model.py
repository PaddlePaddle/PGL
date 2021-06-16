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


class MeanGlobalPool(paddle.nn.Layer):
    def __init__(self, pool_type=None):
        super().__init__()
        self.pool_type = pool_type
        self.pool_fun = gnn.GraphPool("sum")

    def forward(self, graph, nfeat):
        sum_pooled = self.pool_fun(graph, nfeat)
        ones_sum_pooled = self.pool_fun(
            graph, paddle.ones_like(
                nfeat, dtype="float32"))
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
        h = paddle.matmul(paddle.transpose(left_h, [1, 0]), right_h)
        h = paddle.reshape(h, [-1, ])
        return h


class GNN(paddle.nn.Layer):
    def __init__(self, config):
        super(GNN, self).__init__()
        log.info("model_type is %s" % self.__class__.__name__)

        self.config = config
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
                nn.Swish(), L.Linear(hidden_size // 4, self.num_tasks))
        elif self.config.clf_layers == 2:
            log.info("clf_layers is 2")
            self.graph_pred_linear = nn.Sequential(
                L.Linear(hidden_size, hidden_size // 2),
                L.batch_norm_1d(hidden_size // 2),
                nn.Swish(), L.Linear(hidden_size // 2, self.num_tasks))
        else:
            self.graph_pred_linear = L.Linear(hidden_size, self.num_tasks)

    def forward(self, feed_dict):
        h_nodes = []
        for i in range(self.config.block_num):
            h_nodes.append(self.gnn_blocks[i](feed_dict))
        h_node = paddle.concat(h_nodes, axis=-1)

        if self.graph_pooling == "bisop":
            h_graph = self.pool(h_node)
        else:
            h_graph = self.pool(feed_dict['graph'], h_node)
        #  print("graph_repr: ", np.sum(h_graph.numpy()))
        output = self.graph_pred_linear(h_graph)

        #  if self.training:
        #      return output
        #  else:
        #      # At inference time, relu is applied to output to ensure positivity
        #      return paddle.clip(output, min=0, max=50)
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

        self.atom_encoder = getattr(ME, self.config.atom_enc_type,
                                    ME.AtomEncoder)(emb_dim=self.emb_dim)

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
    GNN(num_tasks=10)
