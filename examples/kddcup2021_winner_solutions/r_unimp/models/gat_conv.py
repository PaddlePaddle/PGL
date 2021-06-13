import math
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from pgl.utils.logger import log
import pgl
from pgl.nn import functional as GF

def linear_init(input_size, hidden_size, with_bias=True, init_type='gcn'):
    if init_type == 'gcn':
        fc_w_attr = paddle.ParamAttr(initializer=nn.initializer.XavierNormal())
        fc_bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(0.0))
    else:
        fan_in = input_size
        bias_bound = 1.0 / math.sqrt(fan_in)
        fc_bias_attr = paddle.ParamAttr(initializer=nn.initializer.Uniform(low=-bias_bound, high=bias_bound))

        negative_slope = math.sqrt(5)
        gain = math.sqrt(2.0 / (1 + negative_slope ** 2))
        std = gain / math.sqrt(fan_in)
        weight_bound = math.sqrt(3.0) * std
        fc_w_attr = paddle.ParamAttr(initializer=nn.initializer.Uniform(low=-weight_bound, high=weight_bound))
    
    if not with_bias:
        fc_bias_attr = False
        
    return nn.Linear(input_size, hidden_size, weight_attr=fc_w_attr, bias_attr=fc_bias_attr)


class GATConv(nn.Layer):
    """Implementation of graph attention networks (GAT)
    This is an implementation of the paper GRAPH ATTENTION NETWORKS
    (https://arxiv.org/abs/1710.10903).
    Args:
        input_size: The size of the inputs. 
        hidden_size: The hidden size for gat.
        activation: (default None) The activation for the output.
        num_heads: (default 1) The head number in gat.
        feat_drop: (default 0.6) Dropout rate for feature.
        attn_drop: (default 0.6) Dropout rate for attention.
        concat: (default True) Whether to concat output heads or average them.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 feat_drop=0.6,
                 attn_drop=0.6,
                 num_heads=1,
                 concat=True,
                 activation=None, use_edge=False):
        super(GATConv, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.concat = concat
        if use_edge:
            self.edge_mlp = nn.Sequential(
                nn.Linear(8, 128),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(128, num_heads),
            )
        self.linear = linear_init(input_size, num_heads * hidden_size, init_type='gcn')
        
        fc_w_attr = paddle.ParamAttr(initializer=nn.initializer.XavierNormal())
        self.weight_src = self.create_parameter(shape=[1, num_heads, hidden_size], attr=fc_w_attr)
        
        fc_w_attr = paddle.ParamAttr(initializer=nn.initializer.XavierNormal())
        self.weight_dst = self.create_parameter(shape=[1, num_heads, hidden_size], attr=fc_w_attr)

        self.feat_dropout = nn.Dropout(p=feat_drop)
        self.attn_dropout = nn.Dropout(p=attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        if isinstance(activation, str):
            activation = getattr(F, activation)
        self.activation = activation

    def _send_attention(self, src_feat, dst_feat, edge_feat):
        if "edge_feat" in edge_feat:
            edge_feat = self.edge_mlp(edge_feat["edge_feat"])
            alpha = src_feat["src"] + dst_feat["dst"] + edge_feat
        else:
            alpha = src_feat["src"] + dst_feat["dst"]
        alpha = self.leaky_relu(alpha)
        return {"alpha": alpha, "h": src_feat["h"]}

    def _reduce_attention(self, msg):
        alpha = msg.reduce_softmax(msg["alpha"])
        alpha = paddle.reshape(alpha, [-1, self.num_heads, 1])
        if self.attn_drop > 1e-15:
            alpha = self.attn_dropout(alpha)

        feature = msg["h"]
        feature = paddle.reshape(feature,
                                 [-1, self.num_heads, self.hidden_size])
        feature = feature * alpha
        if self.concat:
            feature = paddle.reshape(feature,
                                     [-1, self.num_heads * self.hidden_size])
        else:
            feature = paddle.mean(feature, axis=1)

        feature = msg.reduce(feature, pool_type="sum")
        return feature

    def forward(self, graph, feature, edge_feat=None):
        """
         
        Args:
 
            graph: `pgl.Graph` instance.
            feature: A tensor with shape (num_nodes, input_size)
     
        Return:
            If `concat=True` then return a tensor with shape (num_nodes, hidden_size),
            else return a tensor with shape (num_nodes, hidden_size * num_heads) 
        """

        if self.feat_drop > 1e-15:
            feature = self.feat_dropout(feature)

        feature = self.linear(feature)
        feature = paddle.reshape(feature,
                                 [-1, self.num_heads, self.hidden_size])

        attn_src = paddle.sum(feature * self.weight_src, axis=-1)
        attn_dst = paddle.sum(feature * self.weight_dst, axis=-1)
        if edge_feat is not None:
            msg = graph.send(
                self._send_attention,
                src_feat={"src": attn_src,
                          "h": feature},
                dst_feat={"dst": attn_dst},
                edge_feat={"edge_feat": edge_feat} )
        else:
            msg = graph.send(
                self._send_attention,
                src_feat={"src": attn_src,
                          "h": feature},
                dst_feat={"dst": attn_dst},
                )
        output = graph.recv(reduce_func=self._reduce_attention, msg=msg)

        if self.activation is not None:
            output = self.activation(output)
        return output
    