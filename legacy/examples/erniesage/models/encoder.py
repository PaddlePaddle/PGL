# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
import json

import numpy as np
import pgl
import paddle.fluid as F
import paddle.fluid.layers as L
from ernie import ErnieModel
from pgl.utils.logger import log


def linear(feat, hidden_size, name, act=None):
    return L.fc(feat,
                hidden_size,
                act=act,
                param_attr=F.ParamAttr(name=name + '_w'),
                bias_attr=F.ParamAttr(name=name + '_b'))


def graphsage_sum(feature, gw, hidden_size, name, act):
    msg = gw.send(lambda s, d, e: s["h"], nfeat_list=[("h", feature)])
    neigh_feature = gw.recv(msg, lambda feat: L.sequence_pool(feat, pool_type="sum"))

    hidden_size = hidden_size
    self_feature = linear(feature, hidden_size, name+"_l", act)
    neigh_feature = linear(neigh_feature, hidden_size, name+"_r", act)
    output = L.concat([self_feature, neigh_feature], axis=1)
    output = L.l2_normalize(output, axis=1)
    return output


class Encoder(object):
    def __init__(self, config):
        self.config = config

    @classmethod
    def factory(cls, config):
        model_type = config.model_type
        if model_type == "ERNIESageV1":
            return ERNIESageV1Encoder(config)
        elif model_type == "ERNIESageV2":
            return ERNIESageV2Encoder(config)
        elif model_type == "ERNIESageV3":
            return ERNIESageV3Encoder(config)
        else:
            raise ValueError

    def __call__(self, graph_wrappers, inputs):
        raise NotImplementedError


class ERNIESageV1Encoder(Encoder):
    def __call__(self, graph_wrappers, inputs):
        feature = self.ernie_pool(graph_wrappers[0].node_feat["term_ids"])

        for i in range(self.config.num_layers):
            feature = graphsage_sum(feature, graph_wrappers[i], self.config.hidden_size, "graphsage_sum_%s"%i, None)

        final_feats = [
            self.take_final_feature(feature, i, "final_fc") for i in inputs
        ]
        return final_feats

    def ernie_pool(self, term_ids):
        cls = L.fill_constant_batch_size_like(term_ids, [-1, 1], "int64",
                                              self.config.cls_id)
        term_ids = L.concat([cls, term_ids], 1)
        ernie_model = ErnieModel(self.config.ernie_config, "")
        feature, _ = ernie_model(term_ids)
        return feature

    def take_final_feature(self, feature, index, name):
        """take final feature"""
        feat = L.gather(feature, index, overwrite=False)

        if self.config.final_fc:
            feat = linear(feat, self.config.hidden_size, name)

        if self.config.final_l2_norm:
            feat = L.l2_normalize(feat, axis=1)
        return feat


class ERNIESageV2Encoder(Encoder):
    def __call__(self, graph_wrappers, inputs):
        feature = graph_wrappers[0].node_feat["term_ids"]
        feature = self.ernie_send_aggregate(graph_wrappers[0], feature,
                                 'leaky_relu', "erniesage_v2")

        for i in range(1, self.config.num_layers):
            feature = graphsage_sum(feature, graph_wrappers[i], self.config.hidden_size, 
                    "graphsage_sum_%s"%i, None)

        final_feats = [
            self.take_final_feature(feature, i, "final_fc") for i in inputs
        ]
        return final_feats

    def take_final_feature(self, feature, index, name):
        """take final feature"""
        feat = L.gather(feature, index, overwrite=False)

        if self.config.final_fc:
            feat = linear(feat, self.config.hidden_size, name)

        if self.config.final_l2_norm:
            feat = L.l2_normalize(feat, axis=1)
        return feat

    def ernie_send_aggregate(self, gw, feature, act, name):

        def ernie_send(src_feat, dst_feat, edge_feat):
            def build_position_ids(term_ids):
                input_mask = L.cast(term_ids > 0, "int64")
                position_ids = L.cumsum(input_mask, axis=1) - 1
                return position_ids
            """doc"""
            # input_ids
            cls = L.fill_constant_batch_size_like(
                src_feat["term_ids"], [-1, 1], "int64", self.config.cls_id)
            src_ids = L.concat([cls, src_feat["term_ids"]], 1)
            dst_ids = dst_feat["term_ids"]

            # sent_ids
            sent_ids = L.concat([L.zeros_like(src_ids), L.ones_like(dst_ids)],
                                1)
            term_ids = L.concat([src_ids, dst_ids], 1)

            # position_ids
            position_ids = build_position_ids(term_ids)
            ernie_model = ErnieModel(self.config.ernie_config, "")
            feature, _ = ernie_model(term_ids, sent_ids, position_ids)
            return feature

        term_ids = feature
        msg = gw.send(ernie_send, nfeat_list=[("term_ids", term_ids)])
        neigh_feature = gw.recv(msg, lambda feat: F.layers.sequence_pool(feat, pool_type="sum"))

        cls = L.fill_constant_batch_size_like(term_ids, [-1, 1],
                                              "int64", self.config.cls_id)
        term_ids = L.concat([cls, term_ids], 1)
        ernie_model = ErnieModel(self.config.ernie_config, "")
        self_feature, _ = ernie_model(term_ids)

        hidden_size = self.config.hidden_size
        self_feature = linear(self_feature, hidden_size, name+"_l", act)
        neigh_feature = linear(neigh_feature, hidden_size, name+"_r", act)
        output = L.concat([self_feature, neigh_feature], axis=1)
        output = L.l2_normalize(output, axis=1)
        return output


class ERNIESageV3Encoder(Encoder):
    def __call__(self, graph_wrappers, inputs):
        feature = graph_wrappers[0].node_feat["term_ids"]
        feature = self.concat_aggregate(graph_wrappers[0], feature, "erniesage_v3_0")

        final_feats = [
            self.take_final_feature(feature, i, "final_fc") for i in inputs
        ]
        return final_feats

    def concat_aggregate(self, gw, feature, name):
        def ernie_recv(feat):
            """doc"""
            num_neighbor = self.config.samples[0]
            pad_value = L.zeros([1], "int64")
            out, _ = L.sequence_pad(
                feat, pad_value=pad_value, maxlen=num_neighbor)
            out = L.reshape(out, [0, self.config.max_seqlen * num_neighbor])
            return out

        msg = gw.send(lambda s, d, e: s["h"], nfeat_list=[("h", feature)])
        neigh_feature = gw.recv(msg, ernie_recv)
        neigh_feature = L.cast(neigh_feature, "int64")

        cls = L.fill_constant_batch_size_like(feature, [-1, 1], "int64",
                                              self.config.cls_id)
        # insert cls, pop last
        term_ids = L.concat([cls, feature[:, :-1], neigh_feature], 1)
        term_ids.stop_gradient = True
        return term_ids

    def take_final_feature(self, feature, index, name):
        """take final feature"""
        term_ids = L.gather(feature, index, overwrite=False)

        ernie_config = self.config.ernie_config
        self.slot_seqlen = self.config.max_seqlen
        position_ids = self._build_position_ids(term_ids)
        sent_ids = self._build_sentence_ids(term_ids)

        ernie_model = ErnieModel(self.config.ernie_config, "")
        feature, _ = ernie_model(term_ids, sent_ids, position_ids)

        if self.config.final_fc:
            feature = linear(feature, self.config.hidden_size, name)

        if self.config.final_l2_norm:
            feature = L.l2_normalize(feature, axis=1)
        return feature

    def _build_position_ids(self, src_ids):
        src_shape = L.shape(src_ids)
        src_seqlen = src_shape[1]
        src_batch = src_shape[0]

        slot_seqlen = self.slot_seqlen

        num_b = (src_seqlen / slot_seqlen) - 1
        a_position_ids = L.reshape(
            L.range(
                0, slot_seqlen, 1, dtype='int32'), [1, slot_seqlen],
            inplace=True)  # [1, slot_seqlen]
        a_position_ids = L.expand(a_position_ids,
                                  [src_batch, 1])  # [B, slot_seqlen]

        input_mask = L.cast(src_ids[:,:slot_seqlen] == 0, "int32")  # assume pad id == 0 [B, slot_seqlen, 1]
        a_pad_len = L.reduce_sum(input_mask, 1)  # [B, 1]

        b_position_ids = L.reshape(
            L.range(
                slot_seqlen, 2 * slot_seqlen, 1, dtype='int32'),
            [1, slot_seqlen],
            inplace=True)  # [1, slot_seqlen]
        b_position_ids = L.expand(
            b_position_ids,
            [src_batch, num_b])  # [B, slot_seqlen * num_b]
        b_position_ids = b_position_ids - a_pad_len  # [B, slot_seqlen * num_b]

        position_ids = L.concat([a_position_ids, b_position_ids], 1)
        position_ids = L.cast(position_ids, 'int64')
        position_ids.stop_gradient = True
        return position_ids

    def _build_sentence_ids(self, src_ids):
        src_shape = L.shape(src_ids)
        src_seqlen = src_shape[1]
        src_batch = src_shape[0]
        slot_seqlen = self.slot_seqlen

        zeros = L.zeros([src_batch, slot_seqlen], "int64")
        ones = L.ones([src_batch, src_seqlen - slot_seqlen], "int64")
        sentence_ids = L.concat([zeros, ones], 1)
        sentence_ids.stop_gradient = True
        return sentence_ids
