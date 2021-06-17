#-*- coding: utf-8 -*-
import numpy as np
import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.distributed as dist

import pgl
import pgl.nn as gnn
from pgl.nn import functional as GF
from pgl.utils.logger import log

import models.mol_encoder as ME

def batch_norm_1d(num_channels):
    if dist.get_world_size() > 1:
        return nn.SyncBatchNorm.convert_sync_batchnorm(nn.BatchNorm1D(num_channels))
    else:
        return nn.BatchNorm1D(num_channels)

class LiteGEMConv(paddle.nn.Layer):
    def __init__(self, config, with_efeat=True):
        super(LiteGEMConv, self).__init__()
        log.info("layer_type is %s" % self.__class__.__name__)
        self.config = config
        self.with_efeat = with_efeat

        self.aggr = self.config.aggr
        self.learn_t = self.config.learn_t
        self.learn_p = self.config.learn_p
        self.init_t = self.config.init_t
        self.init_p = self.config.init_p

        self.eps = 1e-7

        self.emb_dim = self.config.emb_dim

        if self.with_efeat:
            self.bond_encoder = getattr(ME, self.config.bond_enc_type, ME.BondEncoder)(
                    emb_dim = self.emb_dim)

        self.concat = config.concat
        if self.concat:
            self.fc_concat = Linear(self.emb_dim * 3, self.emb_dim)

        assert self.aggr in ['softmax_sg', 'softmax', 'power']

        channels_list = [self.emb_dim]
        for i in range(1, self.config.mlp_layers):
            channels_list.append(self.emb_dim * 2)
        channels_list.append(self.emb_dim)

        self.mlp = MLP(channels_list,
                       norm=self.config.norm,
                       last_lin=True)

        if self.learn_t and self.aggr == "softmax":
            self.t = self.create_parameter(
                shape=[1, ],
                dtype='float32',
                default_initializer=nn.initializer.Constant(value=self.init_t))
        else:
            self.t = self.init_t

        if self.learn_p:
            self.p = self.create_parameter(
                shape=[1, ],
                dtype='float32',
                default_initializer=nn.initializer.Constant(value=self.init_p))

    def send_func(self, src_feat, dst_feat, edge_feat):
        if self.with_efeat:
            if self.concat:
                h = paddle.concat([dst_feat['h'], src_feat['h'], edge_feat['e']], axis=1)
                h = self.fc_concat(h)
            else:
                h = src_feat["h"] + edge_feat["e"]
        else:
            h = src_feat["h"]
        msg = {"h": F.swish(h) + self.eps}
        return msg

    def recv_func(self, msg):
        if self.aggr == "softmax":
            alpha = msg.reduce_softmax(msg["h"] * self.t)
            out = msg['h'] * alpha
            out = msg.reduce_sum(out)
            return out
        elif self.aggr == "power":
            raise NotImplementedError

    def forward(self, graph, nfeat, efeat=None):
        if efeat is not None:
            if self.with_efeat:
                efeat = self.bond_encoder(efeat)

            msg = graph.send(src_feat={"h": nfeat},
                             dst_feat={"h": nfeat},
                             edge_feat={"e": efeat},
                             message_func=self.send_func)
        else:
            msg = graph.send(src_feat={"h": nfeat},
                             dst_feat={"h": nfeat},
                             message_func=self.send_func)

        out = graph.recv(msg=msg, reduce_func=self.recv_func)

        out = nfeat + out
        out = self.mlp(out)

        return out

class CatGINConv(paddle.nn.Layer):
    def __init__(self, config, with_efeat=True):
        super(CatGINConv, self).__init__()
        log.info("layer_type is %s" % self.__class__.__name__)
        self.config = config
        emb_dim = self.config.emb_dim

        self.with_efeat = with_efeat

        self.mlp = nn.Sequential(Linear(emb_dim, emb_dim), 
                                 batch_norm_1d(emb_dim), 
                                 nn.Swish(), 
                                 Linear(emb_dim, emb_dim))

        self.send_mlp = nn.Sequential(nn.Linear(2*emb_dim, 2*emb_dim), 
                                 nn.Swish(), 
                                 Linear(2*emb_dim, emb_dim))

        self.eps = self.create_parameter(
            shape=[1, 1],
            dtype='float32',
            default_initializer=nn.initializer.Constant(value=0))

        if self.with_efeat:
            self.bond_encoder = getattr(ME, self.config.bond_enc_type, ME.BondEncoder)(
                    emb_dim = emb_dim)

    def send_func(self, src_feat, dst_feat, edge_feat):
        if self.with_efeat:
            h = F.relu(src_feat["x"] + edge_feat["e"])
        else:
            h = F.relu(src_feat["x"])
        h = paddle.concat([h, dst_feat['x']], axis=1)
        h = self.send_mlp(h)
        return {"h": h}

    def recv_sum(self, msg):
        return msg.reduce_sum(msg["h"])

    def forward(self, graph, feature, edge_feat=None):

        if self.with_efeat:
            edge_embedding = self.bond_encoder(edge_feat)    

            msg = graph.send(src_feat={"x": feature},
                    dst_feat={"x": feature},
                    edge_feat={"e": edge_embedding},
                    message_func=self.send_func)
        else:
            msg = graph.send(src_feat={"x": feature},
                    dst_feat={"x": feature},
                    message_func=self.send_func)

        neigh_feature = graph.recv(msg=msg, reduce_func=self.recv_sum)

        out = (1 + self.eps) * feature + neigh_feature
        out = self.mlp(out)
        
        return out

class GINConv(paddle.nn.Layer):
    def __init__(self, config, with_efeat=True):
        super(GINConv, self).__init__()
        log.info("layer_type is %s" % self.__class__.__name__)

        self.config = config
        self.with_efeat = with_efeat
        emb_dim = self.config.emb_dim
        self.mlp = nn.Sequential(Linear(emb_dim, emb_dim), 
                                 batch_norm_1d(emb_dim), 
                                 nn.Swish(), 
                                 Linear(emb_dim, emb_dim))
        
        self.eps = self.create_parameter(
            shape=[1, 1],
            dtype='float32',
            default_initializer=nn.initializer.Constant(value=0))

        self.bond_encoder = getattr(ME, self.config.bond_enc_type, ME.BondEncoder)(
                    emb_dim = emb_dim)

    def send_func(self, src_feat, dst_feat, edge_feat):
        return {"h": F.relu(src_feat["x"] + edge_feat["e"])}

    def recv_sum(self, msg):
        return msg.reduce_sum(msg["h"])

    def forward(self, graph, feature, edge_feat):
        edge_embedding = self.bond_encoder(edge_feat)    

        msg = graph.send(src_feat={"x": feature},
                edge_feat={"e": edge_embedding},
                message_func=self.send_func)

        neigh_feature = graph.recv(msg=msg, reduce_func=self.recv_sum)

        out = (1 + self.eps) * feature + neigh_feature
        out = self.mlp(out)
        
        return out

class NormGINConv(paddle.nn.Layer):
    def __init__(self, config, with_efeat=True):
        super(NormGINConv, self).__init__()
        log.info("layer_type is %s" % self.__class__.__name__)

        self.config = config
        self.with_efeat = with_efeat
        emb_dim = self.config.emb_dim
        self.mlp = nn.Sequential(Linear(emb_dim, emb_dim), 
                                 batch_norm_1d(emb_dim), 
                                 nn.Swish(), 
                                 Linear(emb_dim, emb_dim))
        
        self.eps = self.create_parameter(
            shape=[1, 1],
            dtype='float32',
            default_initializer=nn.initializer.Constant(value=0))

        if with_efeat:
            self.bond_encoder = getattr(ME, self.config.bond_enc_type, ME.BondEncoder)(
                        emb_dim = emb_dim)

    def send_func(self, src_feat, dst_feat, edge_feat):
        if self.with_efeat:
            return {"h": F.relu(src_feat["x"] + edge_feat["e"])}
        else:
            return {"h": F.relu(src_feat["x"])}

    def recv_sum(self, msg):
        return msg.reduce_sum(msg["h"])

    def forward(self, graph, feature, edge_feat):
        if self.with_efeat:
            edge_embedding = self.bond_encoder(edge_feat)    

            msg = graph.send(src_feat={"x": feature},
                    edge_feat={"e": edge_embedding},
                    message_func=self.send_func)
        else:
            msg = graph.send(src_feat={"x": feature},
                    dst_feat={"x": feature},
                    message_func=self.send_func)

        neigh_feature = graph.recv(msg=msg, reduce_func=self.recv_sum)

        out = (1 + self.eps) * feature + neigh_feature
        norm = GF.degree_norm(graph)
        out = out * norm
        out = self.mlp(out)
        
        return out

def Linear(input_size, hidden_size, with_bias=True):
    fan_in = input_size
    bias_bound = 1.0 / math.sqrt(fan_in)
    fc_bias_attr = paddle.ParamAttr(initializer=nn.initializer.Uniform(
	low=-bias_bound, high=bias_bound))

    negative_slope = math.sqrt(5)
    gain = math.sqrt(2.0 / (1 + negative_slope**2))
    std = gain / math.sqrt(fan_in)
    weight_bound = math.sqrt(3.0) * std
    fc_w_attr = paddle.ParamAttr(initializer=nn.initializer.Uniform(
	low=-weight_bound, high=weight_bound))

    if not with_bias:
        fc_bias_attr = False

    return nn.Linear(
        input_size, hidden_size, weight_attr=fc_w_attr, bias_attr=fc_bias_attr)

def norm_layer(norm_type, nc):
    # normalization layer 1d
    norm = norm_type.lower()
    if norm == 'batch':
        layer = batch_norm_1d(nc)
    elif norm == 'layer':
        layer = nn.LayerNorm(nc)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm)
    return layer

def act_layer(act_type, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act_type.lower()
    if act == 'relu':
        layer = nn.ReLU()
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'swish':
        layer = nn.Swish()
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

class MLP(paddle.nn.Sequential):
    def __init__(self, channels, act='swish', norm=None, bias=True, drop=0., last_lin=False):
        m = []

        for i in range(1, len(channels)):

            m.append(Linear(channels[i - 1], channels[i], bias))

            if (i == len(channels) - 1) and last_lin:
                pass
            else:
                if norm is not None and norm.lower() != 'none':
                    m.append(norm_layer(norm, channels[i]))
                if act is not None and act.lower() != 'none':
                    m.append(act_layer(act))
                if drop > 0:
                    m.append(nn.Dropout(drop))

        self.m = m
        super(MLP, self).__init__(*self.m)
