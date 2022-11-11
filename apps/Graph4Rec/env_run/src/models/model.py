# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
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
"""Model Definition
"""
import math
import time
from collections import OrderedDict

import paddle
import paddle.nn as nn
import paddle.distributed.fleet as fleet

import pgl
import pgl.nn as gnn
from pgl.distributed import helper
from collections import OrderedDict

import models.embedding as E
import models.layers as GNNLayers

__all__ = ["WalkBasedModel", "SageModel"]


class WalkBasedModel(nn.Layer):
    def __init__(self, config, mode="gpu"):
        super(WalkBasedModel, self).__init__()

        self.config = config
        self.mode = mode
        self.embed_size = self.config.embed_size
        self.hidden_size = self.config.hidden_size
        self.neg_num = self.config.neg_num

        embed_init = nn.initializer.Uniform(
            low=-1. / math.sqrt(self.embed_size),
            high=1. / math.sqrt(self.embed_size))
        emb_attr = paddle.ParamAttr(
            name="node_embedding", initializer=embed_init)

        self.embedding = getattr(E, self.config.embed_type, "BaseEmbedding")(
            self.config.num_nodes,
            self.embed_size,
            sparse=self.config.lazy_mode,
            weight_attr=emb_attr)

        if len(self.config.slots) > 0:
            self.slot_embed_dict = OrderedDict()
            for slot in self.config.slots:
                self.slot_embed_dict[slot] = getattr(
                    E, self.config.embed_type, "BaseEmbedding")(
                        self.config.num_nodes,
                        self.embed_size,
                        sparse=self.config.lazy_mode,
                        weight_attr=paddle.ParamAttr(name="slot_%s_embedding" %
                                                     slot))
            if self.mode == "gpu":
                self.slot_embed_dict = nn.LayerDict(
                    sublayers=self.slot_embed_dict)

        self.loss_fn = paddle.nn.BCEWithLogitsLoss()

    def get_static_input(self):
        """handle static input
        """
        feed_dict = OrderedDict()

        for slot in self.config.slots:
            feed_dict[slot] = paddle.static.data(
                slot, shape=[-1, 1], dtype="int64")
            feed_dict["%s_info" % slot] = paddle.static.data(
                "%s_info" % slot, shape=[-1, ], dtype="int64")

        feed_dict["node_id"] = paddle.static.data(
            "node_id", shape=[-1, 1], dtype="int64")
        feed_dict["neg_idx"] = paddle.static.data(
            "neg_idx", shape=[-1, ], dtype="int64")

        py_reader = paddle.io.DataLoader.from_generator(
            capacity=64,
            feed_list=list(feed_dict.values()),
            iterable=False,
            use_double_buffer=False)

        return feed_dict, py_reader

    def forward(self, feed_dict):
        # [2*b, dim]
        embed = self.embedding(feed_dict['node_id'])

        if len(self.config.slots) > 0:
            nfeat_list = [embed]
            for slot in self.config.slots:
                h = self.slot_embed_dict[slot](feed_dict[slot])
                h = pgl.math.segment_sum(h, feed_dict["%s_info" % slot])
                h = paddle.nn.functional.softsign(h)
                nfeat_list.append(h)
            embed = paddle.stack(nfeat_list, axis=0)
            embed = paddle.sum(embed, axis=0)
            embed = paddle.nn.functional.softsign(embed)

        negs_embed = paddle.gather(embed, feed_dict['neg_idx'])
        embed = paddle.reshape(embed, [-1, 2, self.embed_size])

        src_embed = embed[:, 0:1, :]
        pos_embed = embed[:, 1:2, :]
        negs_embed = paddle.reshape(negs_embed,
                                    [-1, self.neg_num, self.embed_size])

        # [batch_size, 1 + neg_num, embed_size]
        pos_and_neg_embed = paddle.concat([pos_embed, negs_embed], axis=1)

        # [batch_size, 1, 1 + neg_num]
        logits = paddle.matmul(src_embed, pos_and_neg_embed, transpose_y=True)
        # [batch_size, 1 + neg_num]
        logits = paddle.squeeze(logits, axis=1)

        self.embed = paddle.squeeze(src_embed, axis=1)  # for inference
        node_index = paddle.reshape(feed_dict['node_id'], [-1, 2])
        self.node_index = node_index[:, 0]  # for inference

        return logits

    def get_embedding(self):
        return self.node_index, self.embed

    def loss(self, pred, label=None):
        """
            Args:
                pred (Tensor): result of  `self.forward`
                label (Tensor): label can be None if you use unsupervised training
            Returns:
                return (paddle scalar): loss
        """

        pos_logits = pred[:, 0:1]
        neg_logits = pred[:, 1:]

        ones_label = paddle.ones_like(pos_logits)
        zeros_label = paddle.zeros_like(neg_logits)
        pos_loss = self.loss_fn(pos_logits, ones_label)
        neg_loss = self.loss_fn(neg_logits, zeros_label)

        loss = (pos_loss + neg_loss) / 2

        return loss


class SageModel(nn.Layer):
    def __init__(self, config, mode="gpu"):
        super(SageModel, self).__init__()

        self.config = config
        self.mode = mode
        self.embed_size = self.config.embed_size
        self.neg_num = self.config.neg_num
        self.hidden_size = self.config.hidden_size
        self.etype2files = helper.parse_files(self.config.etype2files)
        self.all_etypes = []
        for etype, _ in self.etype2files.items():
            self.all_etypes.append(etype)
            if self.config.symmetry:
                r_etype = helper.get_inverse_etype(etype)
                if r_etype != etype:
                    self.all_etypes.append(r_etype)

        embed_init = nn.initializer.Uniform(
            low=-1. / math.sqrt(self.embed_size),
            high=1. / math.sqrt(self.embed_size))
        emb_attr = paddle.ParamAttr(
            name="node_embedding", initializer=embed_init)

        self.embedding = getattr(E, self.config.embed_type, "BaseEmbedding")(
            self.config.num_nodes,
            self.embed_size,
            sparse=self.config.lazy_mode,
            weight_attr=emb_attr)

        self.loss_fn = paddle.nn.BCEWithLogitsLoss()

        self.convs_dict = OrderedDict()
        for etype in self.all_etypes:
            input_size = self.embed_size
            self.convs_dict[etype] = nn.LayerList()
            for layer in range(len(self.config.sample_num_list)):
                gnn_layer = getattr(GNNLayers, self.config.sage_layer_type)(
                    input_size=input_size,
                    output_size=self.hidden_size,
                    act=self.config.sage_act)
                self.convs_dict[etype].append(gnn_layer)
                input_size = self.hidden_size
        self.convs_dict = nn.LayerDict(sublayers=self.convs_dict)

        if len(self.config.slots) > 0:
            self.slot_embed_dict = OrderedDict()
            for slot in self.config.slots:
                self.slot_embed_dict[slot] = getattr(
                    E, self.config.embed_type, "BaseEmbedding")(
                        self.config.num_nodes,
                        self.embed_size,
                        sparse=self.config.lazy_mode,
                        weight_attr=paddle.ParamAttr(name="slot_%s_embedding" %
                                                     slot))
            if self.mode == "gpu":
                self.slot_embed_dict = nn.LayerDict(
                    sublayers=self.slot_embed_dict)

        if self.config.interact_mode == "gatne":
            self.gatne = GNNLayers.GATNEInteract(self.hidden_size)

    def get_static_input(self):
        """handle static input
        """
        feed_dict = OrderedDict()

        for slot in self.config.slots:
            feed_dict[slot] = paddle.static.data(
                slot, shape=[-1, 1], dtype="int64")
            feed_dict["%s_info" % slot] = paddle.static.data(
                "%s_info" % slot, shape=[-1, ], dtype="int64")

        # Graph 
        if self.config.sage_mode:
            feed_dict["num_nodes"] = paddle.static.data(
                "num_nodes", shape=[1], dtype="int64")

            for etype in self.all_etypes:
                feed_dict["num_edges_%s" % etype] = paddle.static.data(
                    "num_edges_%s" % etype, shape=[1], dtype="int64")
                feed_dict["edges_%s" % etype] = paddle.static.data(
                    "edges_%s" % etype, shape=[-1, 2], dtype="int64")

        feed_dict["origin_node_id"] = paddle.static.data(
            "origin_node_id", shape=[-1, 1], dtype="int64")
        feed_dict["center_node_id"] = paddle.static.data(
            "center_node_id", shape=[-1, ], dtype="int64")

        feed_dict["neg_idx"] = paddle.static.data(
            "neg_idx", shape=[-1], dtype="int64")

        py_reader = paddle.io.DataLoader.from_generator(
            capacity=64,
            feed_list=[v for k, v in feed_dict.items()],
            iterable=False,
            use_double_buffer=False)

        return feed_dict, py_reader

    def forward(self, feed_dict):
        heter_graph = {}
        for etype in self.all_etypes:
            heter_graph[etype] = pgl.Graph(
                num_nodes=feed_dict["num_nodes"],
                num_edges=feed_dict["num_edges_%s" % etype],
                edges=feed_dict["edges_%s" % etype])

        # [b, dim]
        init_nfeat = self.embedding(feed_dict['origin_node_id'])

        if len(self.config.slots) > 0:
            nfeat_list = [init_nfeat]
            for slot in self.config.slots:
                h = self.slot_embed_dict[slot](feed_dict[slot])
                h = pgl.math.segment_sum(h, feed_dict["%s_info" % slot])
                h = paddle.nn.functional.softsign(h)
                nfeat_list.append(h)
            init_nfeat = paddle.stack(nfeat_list, axis=0)
            init_nfeat = paddle.sum(init_nfeat, axis=0)
            init_nfeat = paddle.nn.functional.softsign(init_nfeat)

        nfeat = init_nfeat
        for layer in range(len(self.config.sample_num_list)):
            nxt_fs = []
            for etype, g in heter_graph.items():
                nxt_f = self.convs_dict[etype][layer](heter_graph[etype],
                                                      nfeat)
                nxt_fs.append(nxt_f)

            if self.config.interact_mode == "gatne":
                nfeat = self.gatne(nxt_fs)
            else:
                nfeat = paddle.stack(nxt_fs, axis=1)
                nfeat = paddle.sum(nfeat, axis=1)
            nfeat = init_nfeat * self.config.res_alpha + nfeat * (
                1 - self.config.res_alpha)

        center_nfeat = paddle.gather(nfeat, feed_dict['center_node_id'])
        negs_embed = paddle.gather(center_nfeat, feed_dict['neg_idx'])
        src_and_pos = paddle.reshape(center_nfeat, [-1, 2, self.hidden_size])
        src_embed = src_and_pos[:, 0:1, :]
        pos_embed = src_and_pos[:, 1:2, :]
        negs_embed = paddle.reshape(negs_embed,
                                    [-1, self.neg_num, self.hidden_size])
        pos_and_neg_embed = paddle.concat([pos_embed, negs_embed], axis=1)

        # [batch_size, 1, 1 + neg_num]
        logits = paddle.matmul(src_embed, pos_and_neg_embed, transpose_y=True)
        # [batch_size, 1 + neg_num]
        logits = paddle.squeeze(logits, axis=1)

        # for inference
        self.embed = paddle.squeeze(src_embed, axis=1)
        node_ids = paddle.reshape(feed_dict['origin_node_id'], [-1, 1])
        src_center_idx = paddle.reshape(feed_dict['center_node_id'],
                                        [-1, 2])[:, 0]
        self.node_index = paddle.gather(node_ids, src_center_idx)

        return logits

    def get_embedding(self):
        return self.node_index, self.embed

    def loss(self, pred, label=None):
        """
            Args:
                pred (Tensor): result of  `self.forward`
                label (Tensor): label can be None if you use unsupervised training
            Returns:
                return (paddle scalar): loss
        """

        pos_logits = pred[:, 0:1]
        neg_logits = pred[:, 1:]

        ones_label = paddle.ones_like(pos_logits)
        zeros_label = paddle.zeros_like(neg_logits)
        pos_loss = self.loss_fn(pos_logits, ones_label)
        neg_loss = self.loss_fn(neg_logits, zeros_label)

        loss = (pos_loss + neg_loss) / 2

        return loss
