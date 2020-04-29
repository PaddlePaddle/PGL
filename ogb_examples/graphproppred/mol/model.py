#-*- coding: utf-8 -*-
import os
import re
import time
import logging
from random import random
from functools import reduce, partial

import numpy as np
import multiprocessing

import paddle
import paddle.fluid as F
import paddle.fluid.layers as L
import pgl
from pgl.graph_wrapper import GraphWrapper
from pgl.layers.conv import gcn, gat
from pgl.utils import paddle_helper
from pgl.utils.logger import log

from utils.args import print_arguments, check_cuda, prepare_logger
from utils.init import init_checkpoint, init_pretraining_params

from mol_encoder import AtomEncoder, BondEncoder


def copy_send(src_feat, dst_feat, edge_feat):
    return src_feat["h"]


def mean_recv(feat):
    return L.sequence_pool(feat, pool_type="average")


def sum_recv(feat):
    return L.sequence_pool(feat, pool_type="sum")


def max_recv(feat):
    return L.sequence_pool(feat, pool_type="max")


def unsqueeze(tensor):
    tensor = L.unsqueeze(tensor, axes=-1)
    tensor.stop_gradient = True
    return tensor


class Metric:
    def __init__(self, **args):
        self.args = args

    @property
    def vars(self):
        values = [self.args[k] for k in self.args.keys()]
        return values

    def parse(self, fetch_list):
        tup = list(zip(self.args.keys(), [float(v[0]) for v in fetch_list]))
        return dict(tup)


def gin_layer(gw, node_features, edge_features, train_eps, name):
    def send_func(src_feat, dst_feat, edge_feat):
        """Send"""
        return src_feat["h"] + edge_feat["h"]

    epsilon = L.create_parameter(
        shape=[1, 1],
        dtype="float32",
        attr=F.ParamAttr(name="%s_eps" % name),
        default_initializer=F.initializer.ConstantInitializer(value=0.0))
    if not train_eps:
        epsilon.stop_gradient = True

    msg = gw.send(
        send_func,
        nfeat_list=[("h", node_features)],
        efeat_list=[("h", edge_features)])

    node_feat = gw.recv(msg, "sum") + node_features * (epsilon + 1.0)

    #  if apply_func is not None:
    #      node_feat = apply_func(node_feat, name)
    return node_feat


class GNNModel(object):
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset
        self.hidden_size = self.args.hidden_size
        self.embed_dim = self.args.embed_dim
        self.dropout_prob = self.args.dropout_rate
        self.pool_type = self.args.pool_type
        self._init_vars = []

        graph_data = []
        g, label = self.dataset[0]
        graph_data.append(g)
        g, label = self.dataset[1]
        graph_data.append(g)

        batch_graph = pgl.graph.MultiGraph(graph_data)
        graph_data = batch_graph
        graph_data.edge_feat["feat"] = graph_data.edge_feat["feat"].astype(
            "int64")
        graph_data.node_feat["feat"] = graph_data.node_feat["feat"].astype(
            "int64")
        self.graph_wrapper = GraphWrapper(
            name="graph",
            place=F.CPUPlace(),
            node_feat=graph_data.node_feat_info(),
            edge_feat=graph_data.edge_feat_info())

        self.atom_encoder = AtomEncoder(name="atom", emb_dim=self.embed_dim)
        self.bond_encoder = BondEncoder(name="bond", emb_dim=self.embed_dim)

        self.labels = L.data(
            "labels",
            shape=[None, self.args.num_class],
            dtype="float32",
            append_batch_size=False)

        self.unmask = L.data(
            "unmask",
            shape=[None, self.args.num_class],
            dtype="float32",
            append_batch_size=False)

        self.build_model()

    def build_model(self):
        node_features = self.atom_encoder(self.graph_wrapper.node_feat['feat'])
        edge_features = self.bond_encoder(self.graph_wrapper.edge_feat['feat'])

        self._enc_out = self.node_repr_encode(node_features, edge_features)

        logits = L.fc(self._enc_out,
                      self.args.num_class,
                      act=None,
                      param_attr=F.ParamAttr(name="final_fc"))

        #  L.Print(self.labels, message="labels")
        #  L.Print(self.unmask, message="unmask")
        loss = L.sigmoid_cross_entropy_with_logits(x=logits, label=self.labels)
        loss = loss * self.unmask
        self.loss = L.reduce_sum(loss) / L.reduce_sum(self.unmask)
        self.pred = L.sigmoid(logits)

        self._metrics = Metric(loss=self.loss)

    def node_repr_encode(self, node_features, edge_features):
        features_list = [node_features]
        for layer in range(self.args.num_layers):
            feat = gin_layer(
                self.graph_wrapper,
                features_list[layer],
                edge_features,
                train_eps=self.args.train_eps,
                name="gin_%s" % layer, )

            feat = self.mlp(feat, name="mlp_%s" % layer)

            feat = feat + features_list[layer]  # residual

            features_list.append(feat)

        output = pgl.layers.graph_pooling(
            self.graph_wrapper, features_list[-1], self.args.pool_type)

        return output

    def mlp(self, features, name):
        h = features
        dim = features.shape[-1]
        dim_list = [dim * 2, dim]
        for i in range(2):
            h = L.fc(h,
                     size=dim_list[i],
                     name="%s_fc_%s" % (name, i),
                     act=None)
            if self.args.norm_type == "layer_norm":
                log.info("norm_type is %s" % self.args.norm_type)
                h = L.layer_norm(
                    h,
                    begin_norm_axis=1,
                    param_attr=F.ParamAttr(
                        name="norm_scale_%s_%s" % (name, i),
                        initializer=F.initializer.Constant(1.0)),
                    bias_attr=F.ParamAttr(
                        name="norm_bias_%s_%s" % (name, i),
                        initializer=F.initializer.Constant(0.0)), )
            else:
                log.info("using batch_norm")
                h = L.batch_norm(h)
            h = pgl.layers.graph_norm(self.graph_wrapper, h)
            h = L.relu(h)
        return h

    def get_enc_output(self):
        return self._enc_out

    @property
    def init_vars(self):
        return self._init_vars

    @property
    def metrics(self):
        return self._metrics
