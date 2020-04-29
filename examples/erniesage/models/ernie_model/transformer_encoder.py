#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from functools import partial
import numpy as np
from contextlib import contextmanager

import paddle.fluid as fluid
import paddle.fluid.layers as L
import paddle.fluid.layers as layers
#import propeller.paddle as propeller
#from propeller import log

#determin this at the begining
to_3d = lambda a: a  # will change later
to_2d = lambda a: a


def multi_head_attention(queries,
                         keys,
                         values,
                         attn_bias,
                         d_key,
                         d_value,
                         d_model,
                         n_head=1,
                         dropout_rate=0.,
                         cache=None,
                         param_initializer=None,
                         name='multi_head_att'):
    """
    Multi-Head Attention. Note that attn_bias is added to the logit before
    computing softmax activiation to mask certain selected positions so that
    they will not considered in attention weights.
    """
    keys = queries if keys is None else keys
    values = keys if values is None else values

    def __compute_qkv(queries, keys, values, n_head, d_key, d_value):
        """
        Add linear projection to queries, keys, and values.
        """
        q = layers.fc(input=queries,
                      size=d_key * n_head,
                      num_flatten_dims=len(queries.shape) - 1,
                      param_attr=fluid.ParamAttr(
                          name=name + '_query_fc.w_0',
                          initializer=param_initializer),
                      bias_attr=name + '_query_fc.b_0')
        k = layers.fc(input=keys,
                      size=d_key * n_head,
                      num_flatten_dims=len(keys.shape) - 1,
                      param_attr=fluid.ParamAttr(
                          name=name + '_key_fc.w_0',
                          initializer=param_initializer),
                      bias_attr=name + '_key_fc.b_0')
        v = layers.fc(input=values,
                      size=d_value * n_head,
                      num_flatten_dims=len(values.shape) - 1,
                      param_attr=fluid.ParamAttr(
                          name=name + '_value_fc.w_0',
                          initializer=param_initializer),
                      bias_attr=name + '_value_fc.b_0')
        return q, k, v

    def __split_heads(x, n_head):
        """
        Reshape the last dimension of inpunt tensor x so that it becomes two
        dimensions and then transpose. Specifically, input a tensor with shape
        [bs, max_sequence_length, n_head * hidden_dim] then output a tensor
        with shape [bs, n_head, max_sequence_length, hidden_dim].
        """
        hidden_size = x.shape[-1]
        # The value 0 in shape attr means copying the corresponding dimension
        # size of the input as the output dimension size.
        reshaped = layers.reshape(
            x=x, shape=[0, 0, n_head, hidden_size // n_head], inplace=True)

        # permuate the dimensions into:
        # [batch_size, n_head, max_sequence_len, hidden_size_per_head]
        return layers.transpose(x=reshaped, perm=[0, 2, 1, 3])

    def __combine_heads(x):
        """
        Transpose and then reshape the last two dimensions of inpunt tensor x
        so that it becomes one dimension, which is reverse to __split_heads.
        """
        if len(x.shape) == 3: return x
        if len(x.shape) != 4:
            raise ValueError("Input(x) should be a 4-D Tensor.")
        trans_x = layers.transpose(x, perm=[0, 2, 1, 3])
        # The value 0 in shape attr means copying the corresponding dimension
        # size of the input as the output dimension size.
        #trans_x.desc.set_shape((-1, 1, n_head, d_value))
        return layers.reshape(x=trans_x, shape=[0, 0, d_model], inplace=True)

    def scaled_dot_product_attention(q, k, v, attn_bias, d_key, dropout_rate):
        """
        Scaled Dot-Product Attention
        """
        scaled_q = layers.scale(x=q, scale=d_key**-0.5)
        product = layers.matmul(x=scaled_q, y=k, transpose_y=True)
        if attn_bias:
            product += attn_bias
        weights = layers.softmax(product)
        if dropout_rate:
            weights = layers.dropout(
                weights,
                dropout_prob=dropout_rate,
                dropout_implementation="upscale_in_train",
                is_test=False)
        out = layers.matmul(weights, v)
        #return out, product
        return out, weights

    q, k, v = __compute_qkv(queries, keys, values, n_head, d_key, d_value)
    q = to_3d(q)
    k = to_3d(k)
    v = to_3d(v)

    if cache is not None:  # use cache and concat time steps
        # Since the inplace reshape in __split_heads changes the shape of k and
        # v, which is the cache input for next time step, reshape the cache
        # input from the previous time step first.
        k = cache["k"] = layers.concat(
            [layers.reshape(
                cache["k"], shape=[0, 0, d_model]), k], axis=1)
        v = cache["v"] = layers.concat(
            [layers.reshape(
                cache["v"], shape=[0, 0, d_model]), v], axis=1)

    q = __split_heads(q, n_head)
    k = __split_heads(k, n_head)
    v = __split_heads(v, n_head)

    ctx_multiheads, ctx_multiheads_attn = scaled_dot_product_attention(
        q, k, v, attn_bias, d_key, dropout_rate)

    out = __combine_heads(ctx_multiheads)

    out = to_2d(out)

    # Project back to the model size.
    proj_out = layers.fc(input=out,
                         size=d_model,
                         num_flatten_dims=len(out.shape) - 1,
                         param_attr=fluid.ParamAttr(
                             name=name + '_output_fc.w_0',
                             initializer=param_initializer),
                         bias_attr=name + '_output_fc.b_0')
    return proj_out, ctx_multiheads_attn


def positionwise_feed_forward(x,
                              d_inner_hid,
                              d_hid,
                              dropout_rate,
                              hidden_act,
                              param_initializer=None,
                              name='ffn'):
    """
    Position-wise Feed-Forward Networks.
    This module consists of two linear transformations with a ReLU activation
    in between, which is applied to each position separately and identically.
    """
    hidden = layers.fc(input=x,
                       size=d_inner_hid,
                       num_flatten_dims=len(x.shape) - 1,
                       act=hidden_act,
                       param_attr=fluid.ParamAttr(
                           name=name + '_fc_0.w_0',
                           initializer=param_initializer),
                       bias_attr=name + '_fc_0.b_0')
    if dropout_rate:
        hidden = layers.dropout(
            hidden,
            dropout_prob=dropout_rate,
            dropout_implementation="upscale_in_train",
            is_test=False)
    out = layers.fc(input=hidden,
                    size=d_hid,
                    num_flatten_dims=len(hidden.shape) - 1,
                    param_attr=fluid.ParamAttr(
                        name=name + '_fc_1.w_0',
                        initializer=param_initializer),
                    bias_attr=name + '_fc_1.b_0')
    return out


def pre_post_process_layer(prev_out,
                           out,
                           process_cmd,
                           dropout_rate=0.,
                           name=''):
    """
    Add residual connection, layer normalization and droput to the out tensor
    optionally according to the value of process_cmd.
    This will be used before or after multi-head attention and position-wise
    feed-forward networks.
    """
    for cmd in process_cmd:
        if cmd == "a":  # add residual connection
            out = out + prev_out if prev_out else out
        elif cmd == "n":  # add layer normalization
            out_dtype = out.dtype
            if out_dtype == fluid.core.VarDesc.VarType.FP16:
                out = layers.cast(x=out, dtype="float32")
            out = layers.layer_norm(
                out,
                begin_norm_axis=len(out.shape) - 1,
                param_attr=fluid.ParamAttr(
                    name=name + '_layer_norm_scale',
                    initializer=fluid.initializer.Constant(1.)),
                bias_attr=fluid.ParamAttr(
                    name=name + '_layer_norm_bias',
                    initializer=fluid.initializer.Constant(0.)))
            if out_dtype == fluid.core.VarDesc.VarType.FP16:
                out = layers.cast(x=out, dtype="float16")
        elif cmd == "d":  # add dropout
            if dropout_rate:
                out = layers.dropout(
                    out,
                    dropout_prob=dropout_rate,
                    dropout_implementation="upscale_in_train",
                    is_test=False)
    return out


pre_process_layer = partial(pre_post_process_layer, None)
post_process_layer = pre_post_process_layer


def encoder_layer(enc_input,
                  attn_bias,
                  n_head,
                  d_key,
                  d_value,
                  d_model,
                  d_inner_hid,
                  prepostprocess_dropout,
                  attention_dropout,
                  relu_dropout,
                  hidden_act,
                  preprocess_cmd="n",
                  postprocess_cmd="da",
                  param_initializer=None,
                  name=''):
    """The encoder layers that can be stacked to form a deep encoder.
    This module consits of a multi-head (self) attention followed by
    position-wise feed-forward networks and both the two components companied
    with the post_process_layer to add residual connection, layer normalization
    and droput.
    """
    #L.Print(L.reduce_mean(enc_input), message='1')
    attn_output, ctx_multiheads_attn = multi_head_attention(
        pre_process_layer(
            enc_input,
            preprocess_cmd,
            prepostprocess_dropout,
            name=name + '_pre_att'),
        None,
        None,
        attn_bias,
        d_key,
        d_value,
        d_model,
        n_head,
        attention_dropout,
        param_initializer=param_initializer,
        name=name + '_multi_head_att')
    #L.Print(L.reduce_mean(attn_output), message='1')
    attn_output = post_process_layer(
        enc_input,
        attn_output,
        postprocess_cmd,
        prepostprocess_dropout,
        name=name + '_post_att')

    #L.Print(L.reduce_mean(attn_output), message='2')
    ffd_output = positionwise_feed_forward(
        pre_process_layer(
            attn_output,
            preprocess_cmd,
            prepostprocess_dropout,
            name=name + '_pre_ffn'),
        d_inner_hid,
        d_model,
        relu_dropout,
        hidden_act,
        param_initializer=param_initializer,
        name=name + '_ffn')
    #L.Print(L.reduce_mean(ffd_output), message='3')
    ret = post_process_layer(
        attn_output,
        ffd_output,
        postprocess_cmd,
        prepostprocess_dropout,
        name=name + '_post_ffn')
    #L.Print(L.reduce_mean(ret), message='4')
    return ret, ctx_multiheads_attn, ffd_output


def build_pad_idx(input_mask):
    pad_idx = L.where(L.cast(L.squeeze(input_mask, [2]), 'bool'))
    return pad_idx


def build_attn_bias(input_mask, n_head, dtype):
    attn_bias = L.matmul(
        input_mask, input_mask, transpose_y=True)  # [batch, seq, seq]
    attn_bias = (1. - attn_bias) * -10000.
    attn_bias = L.stack([attn_bias] * n_head, 1) # [batch, n_head, seq, seq]
    if attn_bias.dtype != dtype:
        attn_bias = L.cast(attn_bias, dtype)
    return attn_bias


def build_graph_attn_bias(input_mask, n_head, dtype, slot_seqlen):

    input_shape = L.shape(input_mask)
    input_batch = input_shape[0]
    input_seqlen = input_shape[1]
    num_slot = input_seqlen / slot_seqlen
    num_b = num_slot - 1
    ones = L.ones([num_b], dtype="float32") # [num_b]
    diag_ones = L.diag(ones) # [num_b, num_b]
    diag_ones = L.unsqueeze(diag_ones, [1, -1]) # [num_b, 1, num_b, 1]
    diag_ones = L.expand(diag_ones, [1, slot_seqlen, 1, slot_seqlen]) # [num_b, seqlen, num_b, seqlen]
    diag_ones = L.reshape(diag_ones, [1, num_b*slot_seqlen, num_b*slot_seqlen]) # [1, num_b*seqlen, num_b*seqlen]
    
    graph_attn_bias = L.concat([L.ones([1, num_b*slot_seqlen, slot_seqlen], dtype="float32"), diag_ones], 2)
    graph_attn_bias = L.concat([L.ones([1, slot_seqlen, num_slot*slot_seqlen], dtype="float32"), graph_attn_bias], 1) # [1, seq, seq]

    pad_attn_bias = L.matmul(
        input_mask, input_mask, transpose_y=True)  # [batch, seq, seq]
    attn_bias = graph_attn_bias * pad_attn_bias

    attn_bias = (1. - attn_bias) * -10000.
    attn_bias = L.stack([attn_bias] * n_head, 1) # [batch, n_head, seq, seq]
    if attn_bias.dtype != dtype:
        attn_bias = L.cast(attn_bias, dtype)
    return attn_bias


def encoder(enc_input,
            input_mask,
            n_layer,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            prepostprocess_dropout,
            attention_dropout,
            relu_dropout,
            hidden_act,
            preprocess_cmd="n",
            postprocess_cmd="da",
            param_initializer=None,
            name=''):
    """
    The encoder is composed of a stack of identical layers returned by calling
    encoder_layer.
    """

    #global to_2d, to_3d  #, batch, seqlen, dynamic_dim
    d_shape = L.shape(input_mask)
    pad_idx = build_pad_idx(input_mask)
    attn_bias = build_attn_bias(input_mask, n_head, enc_input.dtype)

    # d_batch = d_shape[0]
    # d_seqlen = d_shape[1]
    # pad_idx = L.where(
    # L.cast(L.reshape(input_mask, [d_batch, d_seqlen]), 'bool'))

    # attn_bias = L.matmul(
    # input_mask, input_mask, transpose_y=True)  # [batch, seq, seq]
    # attn_bias = (1. - attn_bias) * -10000.
    # attn_bias = L.stack([attn_bias] * n_head, 1)
    # if attn_bias.dtype != enc_input.dtype:
    # attn_bias = L.cast(attn_bias, enc_input.dtype)

    # def to_2d(t_3d):
        # t_2d = L.gather_nd(t_3d, pad_idx)
        # return t_2d

    # def to_3d(t_2d):
        # t_3d = L.scatter_nd(
        # pad_idx, t_2d, shape=[d_shape[0], d_shape[1], d_model])
        # return t_3d

    enc_input = to_2d(enc_input)
    all_hidden = []
    all_attn = []
    all_ffn = []
    for i in range(n_layer):
        enc_output, ctx_multiheads_attn, ffn_output = encoder_layer(
            enc_input,
            attn_bias,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            prepostprocess_dropout,
            attention_dropout,
            relu_dropout,
            hidden_act,
            preprocess_cmd,
            postprocess_cmd,
            param_initializer=param_initializer,
            name=name + '_layer_' + str(i))
        all_hidden.append(enc_output)
        all_attn.append(ctx_multiheads_attn)
        all_ffn.append(ffn_output)
        enc_input = enc_output
    enc_output = pre_process_layer(
        enc_output,
        preprocess_cmd,
        prepostprocess_dropout,
        name="post_encoder")
    enc_output = to_3d(enc_output)
    #enc_output.desc.set_shape((-1, 1, final_dim))
    return enc_output, all_hidden, all_attn, all_ffn

def graph_encoder(enc_input,
            input_mask,
            n_layer,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            prepostprocess_dropout,
            attention_dropout,
            relu_dropout,
            hidden_act,
            preprocess_cmd="n",
            postprocess_cmd="da",
            param_initializer=None,
            slot_seqlen=40,
            name=''):
    """
    The encoder is composed of a stack of identical layers returned by calling
    encoder_layer.
    """

    #global to_2d, to_3d  #, batch, seqlen, dynamic_dim
    d_shape = L.shape(input_mask)
    pad_idx = build_pad_idx(input_mask)
    attn_bias = build_graph_attn_bias(input_mask, n_head, enc_input.dtype, slot_seqlen)
    #attn_bias = build_attn_bias(input_mask, n_head, enc_input.dtype)

    # d_batch = d_shape[0]
    # d_seqlen = d_shape[1]
    # pad_idx = L.where(
    # L.cast(L.reshape(input_mask, [d_batch, d_seqlen]), 'bool'))

    # attn_bias = L.matmul(
    # input_mask, input_mask, transpose_y=True)  # [batch, seq, seq]
    # attn_bias = (1. - attn_bias) * -10000.
    # attn_bias = L.stack([attn_bias] * n_head, 1)
    # if attn_bias.dtype != enc_input.dtype:
    # attn_bias = L.cast(attn_bias, enc_input.dtype)

    # def to_2d(t_3d):
        # t_2d = L.gather_nd(t_3d, pad_idx)
        # return t_2d

    # def to_3d(t_2d):
        # t_3d = L.scatter_nd(
        # pad_idx, t_2d, shape=[d_shape[0], d_shape[1], d_model])
        # return t_3d

    enc_input = to_2d(enc_input)
    all_hidden = []
    all_attn = []
    all_ffn = []
    for i in range(n_layer):
        enc_output, ctx_multiheads_attn, ffn_output = encoder_layer(
            enc_input,
            attn_bias,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            prepostprocess_dropout,
            attention_dropout,
            relu_dropout,
            hidden_act,
            preprocess_cmd,
            postprocess_cmd,
            param_initializer=param_initializer,
            name=name + '_layer_' + str(i))
        all_hidden.append(enc_output)
        all_attn.append(ctx_multiheads_attn)
        all_ffn.append(ffn_output)
        enc_input = enc_output
    enc_output = pre_process_layer(
        enc_output,
        preprocess_cmd,
        prepostprocess_dropout,
        name="post_encoder")
    enc_output = to_3d(enc_output)
    #enc_output.desc.set_shape((-1, 1, final_dim))
    return enc_output, all_hidden, all_attn, all_ffn
