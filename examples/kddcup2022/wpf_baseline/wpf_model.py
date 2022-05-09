# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import pgl
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from pgl.utils.logger import log
import pgl.nn as gnn
import math

WIN = 3
DECOMP = 24


class SeriesDecomp(nn.Layer):
    """Ideas comes from AutoFormer
    Decompose a time series into trends and seasonal

    Refs:  https://arxiv.org/abs/2106.13008
    """

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        t_x = paddle.transpose(x, [0, 2, 1])
        mean_x = F.avg_pool1d(
            t_x, self.kernel_size, stride=1, padding="SAME", exclusive=False)
        mean_x = paddle.transpose(mean_x, [0, 2, 1])
        return x - mean_x, mean_x


class TransformerDecoderLayer(nn.Layer):
    """Transformer Decoder with Time Series Decomposition

    Ideas comes from AutoFormer

    Decoding trends and seasonal 
    Decompose a time series into trends and seasonal

    Refs:  https://arxiv.org/abs/2106.13008
    """

    def __init__(self,
                 d_model,
                 nhead,
                 dims_feedforward,
                 dropout=0.1,
                 activation="gelu",
                 attn_dropout=None,
                 act_dropout=None,
                 trends_out=134,
                 weight_attr=None,
                 bias_attr=None):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3

        super(TransformerDecoderLayer, self).__init__()

        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout

        self.decomp = SeriesDecomp(DECOMP)

        self.self_attn = nn.MultiHeadAttention(d_model, nhead)

        self.cross_attn = nn.MultiHeadAttention(d_model, nhead)

        self.linear1 = nn.Linear(d_model, dims_feedforward)
        self.dropout = nn.Dropout(act_dropout, mode="upscale_in_train")
        self.linear2 = nn.Linear(dims_feedforward, d_model)

        self.linear_trend = nn.Conv1D(
            d_model, trends_out, WIN, padding="SAME", data_format="NLC")

        self.dropout1 = nn.Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = nn.Dropout(dropout, mode="upscale_in_train")
        self.activation = getattr(F, activation)

    def forward(self, src, memory, src_mask=None, cache=None):
        residual = src
        src = self.self_attn(src, src, src, None)
        src = residual + self.dropout1(src)

        src, trend1 = self.decomp(src)

        residual = src
        src = self.self_attn(src, memory, memory, None)
        src = residual + self.dropout1(src)

        src, trend2 = self.decomp(src)
        #    pass

        residual = src

        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)

        src, trend3 = self.decomp(src)
        res_trend = trend1 + trend2 + trend3
        res_trend = self.linear_trend(res_trend)
        return src, res_trend


class TransformerEncoderLayer(nn.Layer):
    """Transformer Encoder with Time Series Decomposition

    Ideas comes from AutoFormer

    Decoding trends and seasonal 
    Decompose a time series into trends and seasonal

    Refs:  https://arxiv.org/abs/2106.13008
    """

    def __init__(self,
                 d_model,
                 nhead,
                 dims_feedforward,
                 dropout=0.1,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 weight_attr=None,
                 bias_attr=None):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3

        super(TransformerEncoderLayer, self).__init__()

        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout

        self.decomp = SeriesDecomp(DECOMP)

        self.self_attn = nn.MultiHeadAttention(d_model, nhead)

        self.linear1 = nn.Linear(d_model, dims_feedforward)
        self.dropout = nn.Dropout(act_dropout, mode="upscale_in_train")
        self.linear2 = nn.Linear(dims_feedforward, d_model)

        self.dropout1 = nn.Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = nn.Dropout(dropout, mode="upscale_in_train")
        self.activation = getattr(F, activation)

    def forward(self, src, src_mask=None, cache=None):
        residual = src
        src = self.self_attn(src, src, src, None)
        src = residual + self.dropout1(src)

        src, _ = self.decomp(src)

        residual = src

        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)

        src, _ = self.decomp(src)
        return src


class Encoder(nn.Layer):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config_file = config
        self.var_len = config.var_len
        self.input_len = config.input_len
        self.output_len = config.output_len
        self.hidden_dims = config.model.hidden_dims
        self.nhead = config.model.nhead
        self.num_encoder_layer = config.model.encoder_layers

        self.enc_lins = nn.LayerList()
        self.dropout = config.model.dropout
        self.drop = nn.Dropout(self.dropout)
        for _ in range(self.num_encoder_layer):
            self.enc_lins.append(
                TransformerEncoderLayer(
                    d_model=self.hidden_dims,
                    nhead=self.nhead,
                    dropout=self.dropout,
                    activation="gelu",
                    attn_dropout=self.dropout,
                    act_dropout=self.dropout,
                    dims_feedforward=self.hidden_dims * 2))

    def forward(self, batch_x):
        for lin in self.enc_lins:
            batch_x = lin(batch_x)
        batch_x = self.drop(batch_x)
        return batch_x


class Decoder(nn.Layer):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config_file = config
        self.var_len = config.var_len
        self.input_len = config.input_len
        self.output_len = config.output_len
        self.hidden_dims = config.model.hidden_dims
        self.nhead = config.model.nhead
        self.num_decoder_layer = config.model.decoder_layers

        self.dec_lins = nn.LayerList()
        self.dropout = config.model.dropout
        self.drop = nn.Dropout(self.dropout)
        self.capacity = config.capacity

        for _ in range(self.num_decoder_layer):
            self.dec_lins.append(
                TransformerDecoderLayer(
                    d_model=self.hidden_dims,
                    nhead=self.nhead,
                    dropout=self.dropout,
                    activation="gelu",
                    attn_dropout=self.dropout,
                    act_dropout=self.dropout,
                    dims_feedforward=self.hidden_dims * 2,
                    trends_out=self.capacity))

    def forward(self, season, trend, enc_output):
        for lin in self.dec_lins:
            season, trend_part = lin(season, enc_output)
            trend = trend + trend_part
        return season, trend


class SpatialTemporalConv(nn.Layer):
    """ Spatial Temporal Embedding

    Apply GAT and Conv1D based on Temporal and Spatial Correlation
    """

    def __init__(self, id_len, input_dim, output_dim):
        super(SpatialTemporalConv, self).__init__()
        self.conv1 = nn.Conv1D(
            id_len * input_dim,
            output_dim,
            kernel_size=WIN,
            padding="SAME",
            data_format="NLC",
            bias_attr=False)
        self.id_len = id_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.q = nn.Linear(input_dim, output_dim)
        self.k = nn.Linear(input_dim, output_dim)

    def _send_attention(self, src_feat, dst_feat, edge_feat):
        alpha = src_feat["k"] * dst_feat["q"]
        alpha = paddle.sum(alpha, -1, keepdim=True)
        return {"alpha": alpha, "output_series": src_feat["v"]}

    def _reduce_attention(self, msg):
        alpha = msg.reduce_softmax(msg["alpha"])
        return msg.reduce(msg["output_series"] * alpha, pool_type="sum")

    def forward(self, x, graph):
        bz, seqlen, _ = x.shape
        x = paddle.reshape(x, [bz, seqlen, self.id_len, self.input_dim])
        x = paddle.transpose(x, [0, 2, 1, 3])
        x = paddle.reshape(x, [-1, seqlen, self.input_dim])
        mean_x = paddle.mean(x, 1)
        q_x = self.q(mean_x) / math.sqrt(self.output_dim)
        k_x = self.k(mean_x)
        x = paddle.reshape(x, [-1, seqlen * self.input_dim])

        msg = graph.send(
            self._send_attention,
            src_feat={"k": k_x,
                      "v": x},
            dst_feat={"q": q_x})
        output = graph.recv(reduce_func=self._reduce_attention, msg=msg)
        x = paddle.reshape(output, [bz, self.id_len, seqlen, self.input_dim])
        x = paddle.transpose(x, [0, 2, 1, 3])
        x = paddle.reshape(x, [bz, seqlen, self.id_len * self.input_dim])
        return self.conv1(x)


class WPFModel(nn.Layer):
    """Models for Wind Power Prediction
    """

    def __init__(self, config):
        super(WPFModel, self).__init__()
        self.config_file = config
        self.var_len = config.var_len
        self.input_len = config.input_len
        self.output_len = config.output_len
        self.hidden_dims = config.model.hidden_dims
        self.capacity = config.capacity

        self.decomp = SeriesDecomp(DECOMP)

        self.t_emb = nn.Embedding(300, self.hidden_dims)
        self.w_emb = nn.Embedding(300, self.hidden_dims)

        self.t_dec_emb = nn.Embedding(300, self.hidden_dims)
        self.w_dec_emb = nn.Embedding(300, self.hidden_dims)

        self.pos_dec_emb = paddle.create_parameter(
            shape=[1, self.input_len + self.output_len, self.hidden_dims],
            dtype='float32')

        self.pos_emb = paddle.create_parameter(
            shape=[1, self.input_len, self.hidden_dims], dtype='float32')

        self.st_conv_encoder = SpatialTemporalConv(self.capacity, self.var_len,
                                                   self.hidden_dims)
        self.st_conv_decoder = SpatialTemporalConv(self.capacity, self.var_len,
                                                   self.hidden_dims)

        self.enc = Encoder(config)
        self.dec = Decoder(config)

        self.pred_nn = nn.Linear(self.hidden_dims, self.capacity)
        self.apply(self.init_weights)

    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0, std=0.02, shape=layer.weight.shape))

        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-12

    def forward(self, batch_x, batch_y, data_mean, data_scale, graph=None):
        bz, id_len, input_len, var_len = batch_x.shape

        batch_graph = pgl.Graph.batch([graph] * bz)

        _, _, output_len, _ = batch_y.shape
        var_len = var_len - 2

        time_id = batch_x[:, 0, :, 1].astype("int32")
        weekday_id = batch_x[:, 0, :, 0].astype("int32")

        batch_x = batch_x[:, :, :, 2:]
        batch_x = (batch_x - data_mean) / data_scale

        y_weekday_id = batch_y[:, 0, :, 0].astype("int32")
        y_time_id = batch_y[:, 0, :, 1].astype("int32")

        batch_x_time_emb = self.t_emb(time_id)
        batch_y_time_emb = self.t_dec_emb(
            paddle.concat([time_id, y_time_id], 1))

        batch_x_weekday_emb = self.w_emb(weekday_id)
        batch_y_weekday_emb = self.w_dec_emb(
            paddle.concat([weekday_id, y_weekday_id], 1))

        batch_x = paddle.transpose(batch_x, [0, 2, 1, 3])

        batch_pred_trend = paddle.mean(batch_x, 1, keepdim=True)[:, :, :, -1]
        batch_pred_trend = paddle.tile(batch_pred_trend, [1, output_len, 1])
        batch_pred_trend = paddle.concat(
            [self.decomp(batch_x[:, :, :, -1])[0], batch_pred_trend], 1)

        batch_x = paddle.reshape(batch_x, [bz, input_len, var_len * id_len])
        _, season_init = self.decomp(batch_x)

        batch_pred_season = paddle.zeros(
            [bz, output_len, var_len * id_len], dtype="float32")
        batch_pred_season = paddle.concat([season_init, batch_pred_season], 1)

        batch_x = self.st_conv_encoder(batch_x, batch_graph) + self.pos_emb

        batch_pred_season = self.st_conv_decoder(
            batch_pred_season, batch_graph) + self.pos_dec_emb

        batch_x = self.enc(batch_x)

        batch_x_pred, batch_x_trends = self.dec(batch_pred_season,
                                                batch_pred_trend, batch_x)
        batch_x_pred = self.pred_nn(batch_x_pred)

        pred_y = batch_x_pred + batch_x_trends
        pred_y = paddle.transpose(pred_y, [0, 2, 1])
        pred_y = pred_y[:, :, -output_len:]
        return pred_y
