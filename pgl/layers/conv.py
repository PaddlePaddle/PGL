# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
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
"""This package implements common layers to help building
graph neural networks.
"""
import paddle.fluid as fluid
from pgl import graph_wrapper
from pgl.utils import paddle_helper

__all__ = ['gcn', 'gat', 'gin']


def gcn(gw, feature, hidden_size, activation, name, norm=None):
    """Implementation of graph convolutional neural networks (GCN)

    This is an implementation of the paper SEMI-SUPERVISED CLASSIFICATION
    WITH GRAPH CONVOLUTIONAL NETWORKS (https://arxiv.org/pdf/1609.02907.pdf).

    Args:
        gw: Graph wrapper object (:code:`StaticGraphWrapper` or :code:`GraphWrapper`)

        feature: A tensor with shape (num_nodes, feature_size).

        hidden_size: The hidden size for gcn.

        activation: The activation for the output.

        name: Gcn layer names.

        norm: If :code:`norm` is not None, then the feature will be normalized. Norm must
              be tensor with shape (num_nodes,) and dtype float32.

    Return:
        A tensor with shape (num_nodes, hidden_size)
    """

    def send_src_copy(src_feat, dst_feat, edge_feat):
        return src_feat["h"]

    size = feature.shape[-1]
    if size > hidden_size:
        feature = fluid.layers.fc(feature,
                                  size=hidden_size,
                                  bias_attr=False,
                                  param_attr=fluid.ParamAttr(name=name))

    if norm is not None:
        feature = feature * norm

    msg = gw.send(send_src_copy, nfeat_list=[("h", feature)])

    if size > hidden_size:
        output = gw.recv(msg, "sum")
    else:
        output = gw.recv(msg, "sum")
        output = fluid.layers.fc(output,
                                 size=hidden_size,
                                 bias_attr=False,
                                 param_attr=fluid.ParamAttr(name=name))

    if norm is not None:
        output = output * norm

    bias = fluid.layers.create_parameter(
        shape=[hidden_size],
        dtype='float32',
        is_bias=True,
        name=name + '_bias')
    output = fluid.layers.elementwise_add(output, bias, act=activation)
    return output


def gat(gw,
        feature,
        hidden_size,
        activation,
        name,
        num_heads=8,
        feat_drop=0.6,
        attn_drop=0.6,
        is_test=False):
    """Implementation of graph attention networks (GAT)

    This is an implementation of the paper GRAPH ATTENTION NETWORKS
    (https://arxiv.org/abs/1710.10903).

    Args:
        gw: Graph wrapper object (:code:`StaticGraphWrapper` or :code:`GraphWrapper`)

        feature: A tensor with shape (num_nodes, feature_size).

        hidden_size: The hidden size for gat.

        activation: The activation for the output.

        name: Gat layer names.

        num_heads: The head number in gat.

        feat_drop: Dropout rate for feature.

        attn_drop: Dropout rate for attention.

        is_test: Whether in test phrase.

    Return:
        A tensor with shape (num_nodes, hidden_size * num_heads)
    """

    def send_attention(src_feat, dst_feat, edge_feat):
        output = src_feat["left_a"] + dst_feat["right_a"]
        output = fluid.layers.leaky_relu(
            output, alpha=0.2)  # (num_edges, num_heads)
        return {"alpha": output, "h": src_feat["h"]}

    def reduce_attention(msg):
        alpha = msg["alpha"]  # lod-tensor (batch_size, seq_len, num_heads)
        h = msg["h"]
        alpha = paddle_helper.sequence_softmax(alpha)
        old_h = h
        h = fluid.layers.reshape(h, [-1, num_heads, hidden_size])
        alpha = fluid.layers.reshape(alpha, [-1, num_heads, 1])
        if attn_drop > 1e-15:
            alpha = fluid.layers.dropout(
                alpha,
                dropout_prob=attn_drop,
                is_test=is_test,
                dropout_implementation="upscale_in_train")
        h = h * alpha
        h = fluid.layers.reshape(h, [-1, num_heads * hidden_size])
        h = fluid.layers.lod_reset(h, old_h)
        return fluid.layers.sequence_pool(h, "sum")

    if feat_drop > 1e-15:
        feature = fluid.layers.dropout(
            feature,
            dropout_prob=feat_drop,
            is_test=is_test,
            dropout_implementation='upscale_in_train')

    ft = fluid.layers.fc(feature,
                         hidden_size * num_heads,
                         bias_attr=False,
                         param_attr=fluid.ParamAttr(name=name + '_weight'))
    left_a = fluid.layers.create_parameter(
        shape=[num_heads, hidden_size],
        dtype='float32',
        name=name + '_gat_l_A')
    right_a = fluid.layers.create_parameter(
        shape=[num_heads, hidden_size],
        dtype='float32',
        name=name + '_gat_r_A')
    reshape_ft = fluid.layers.reshape(ft, [-1, num_heads, hidden_size])
    left_a_value = fluid.layers.reduce_sum(reshape_ft * left_a, -1)
    right_a_value = fluid.layers.reduce_sum(reshape_ft * right_a, -1)

    msg = gw.send(
        send_attention,
        nfeat_list=[("h", ft), ("left_a", left_a_value),
                    ("right_a", right_a_value)])
    output = gw.recv(msg, reduce_attention)
    bias = fluid.layers.create_parameter(
        shape=[hidden_size * num_heads],
        dtype='float32',
        is_bias=True,
        name=name + '_bias')
    bias.stop_gradient = True
    output = fluid.layers.elementwise_add(output, bias, act=activation)
    return output


def gin(gw,
        feature,
        hidden_size,
        activation,
        name,
        init_eps=0.0,
        train_eps=False):
    """Implementation of Graph Isomorphism Network (GIN) layer.

    This is an implementation of the paper How Powerful are Graph Neural Networks?
    (https://arxiv.org/pdf/1810.00826.pdf).

    In their implementation, all MLPs have 2 layers. Batch normalization is applied
    on every hidden layer.

    Args:
        gw: Graph wrapper object (:code:`StaticGraphWrapper` or :code:`GraphWrapper`)

        feature: A tensor with shape (num_nodes, feature_size).

        name: GIN layer names.

        hidden_size: The hidden size for gin.

        activation: The activation for the output.

        init_eps: float, optional
            Initial :math:`\epsilon` value, default is 0.

        train_eps: bool, optional
            if True, :math:`\epsilon` will be a learnable parameter.

    Return:
        A tensor with shape (num_nodes, hidden_size).
    """

    def send_src_copy(src_feat, dst_feat, edge_feat):
        return src_feat["h"]

    epsilon = fluid.layers.create_parameter(
        shape=[1, 1],
        dtype="float32",
        attr=fluid.ParamAttr(name="%s_eps" % name),
        default_initializer=fluid.initializer.ConstantInitializer(
            value=init_eps))

    if not train_eps:
        epsilon.stop_gradient = True

    msg = gw.send(send_src_copy, nfeat_list=[("h", feature)])
    output = gw.recv(msg, "sum") + feature * (epsilon + 1.0)

    output = fluid.layers.fc(output,
                             size=hidden_size,
                             act=None,
                             param_attr=fluid.ParamAttr(name="%s_w_0" % name),
                             bias_attr=fluid.ParamAttr(name="%s_b_0" % name))

    output = fluid.layers.layer_norm(
        output,
        begin_norm_axis=1,
        param_attr=fluid.ParamAttr(
            name="norm_scale_%s" % (name),
            initializer=fluid.initializer.Constant(1.0)),
        bias_attr=fluid.ParamAttr(
            name="norm_bias_%s" % (name),
            initializer=fluid.initializer.Constant(0.0)), )

    if activation is not None:
        output = getattr(fluid.layers, activation)(output)

    output = fluid.layers.fc(output,
                             size=hidden_size,
                             act=activation,
                             param_attr=fluid.ParamAttr(name="%s_w_1" % name),
                             bias_attr=fluid.ParamAttr(name="%s_b_1" % name))

    return output
