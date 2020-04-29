#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""Ernie model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import six
import logging
import paddle.fluid as fluid
import paddle.fluid.layers as L

from io import open

from models.ernie_model.transformer_encoder import encoder, pre_process_layer
from models.ernie_model.transformer_encoder import graph_encoder

log = logging.getLogger(__name__)


class ErnieConfig(object):
    def __init__(self, config_path):
        self._config_dict = self._parse(config_path)

    def _parse(self, config_path):
        try:
            with open(config_path, 'r', encoding='utf8') as json_file:
                config_dict = json.load(json_file)
        except Exception:
            raise IOError("Error in parsing Ernie model config file '%s'" %
                          config_path)
        else:
            return config_dict

    def __getitem__(self, key):
        return self._config_dict.get(key, None)

    def print_config(self):
        for arg, value in sorted(six.iteritems(self._config_dict)):
            log.info('%s: %s' % (arg, value))
        log.info('------------------------------------------------')


class ErnieModel(object):
    def __init__(self,
                 src_ids,
                 sentence_ids,
                 task_ids=None,
                 config=None,
                 weight_sharing=True,
                 use_fp16=False,
                 name=""):

        self._set_config(config, name, weight_sharing)
        input_mask = self._build_input_mask(src_ids)
        position_ids = self._build_position_ids(src_ids)
        self._build_model(src_ids, position_ids, sentence_ids, task_ids,
                          input_mask)
        self._debug_summary(input_mask)

    def _debug_summary(self, input_mask):
        #histogram
        seqlen_before_pad = L.cast(
            L.reduce_sum(
                input_mask, dim=1), dtype='float32')
        seqlen_after_pad = L.reduce_sum(
            L.cast(
                L.zeros_like(input_mask), dtype='float32') + 1.0, dim=1)
        pad_num = seqlen_after_pad - seqlen_before_pad
        pad_rate = pad_num / seqlen_after_pad

    def _build_position_ids(self, src_ids):
        d_shape = L.shape(src_ids)
        d_seqlen = d_shape[1]
        d_batch = d_shape[0]
        position_ids = L.reshape(
            L.range(
                0, d_seqlen, 1, dtype='int32'), [1, d_seqlen, 1],
            inplace=True)
        position_ids = L.expand(position_ids, [d_batch, 1, 1])
        position_ids = L.cast(position_ids, 'int64')
        position_ids.stop_gradient = True
        return position_ids

    def _build_input_mask(self, src_ids):
        zero = L.fill_constant([1], dtype='int64', value=0)
        input_mask = L.logical_not(L.equal(src_ids,
                                           zero))  # assume pad id == 0
        input_mask = L.cast(input_mask, 'float')
        input_mask.stop_gradient = True
        return input_mask

    def _set_config(self, config, name, weight_sharing):
        self._emb_size = config['hidden_size']
        self._n_layer = config['num_hidden_layers']
        self._n_head = config['num_attention_heads']
        self._voc_size = config['vocab_size']
        self._max_position_seq_len = config['max_position_embeddings']
        if config.get('sent_type_vocab_size'):
            self._sent_types = config['sent_type_vocab_size']
        else:
            self._sent_types = config['type_vocab_size']

        self._use_task_id = config['use_task_id']
        if self._use_task_id:
            self._task_types = config['task_type_vocab_size']
        self._hidden_act = config['hidden_act']
        self._postprocess_cmd = config.get('postprocess_cmd', 'dan')
        self._preprocess_cmd = config.get('preprocess_cmd', '')
        self._prepostprocess_dropout = config['hidden_dropout_prob']
        self._attention_dropout = config['attention_probs_dropout_prob']
        self._weight_sharing = weight_sharing
        self.name = name

        self._word_emb_name = self.name + "word_embedding"
        self._pos_emb_name = self.name + "pos_embedding"
        self._sent_emb_name = self.name + "sent_embedding"
        self._task_emb_name = self.name + "task_embedding"
        self._dtype = "float16" if config['use_fp16'] else "float32"
        self._emb_dtype = "float32"

        # Initialize all weigths by truncated normal initializer, and all biases
        # will be initialized by constant zero by default.
        self._param_initializer = fluid.initializer.TruncatedNormal(
            scale=config['initializer_range'])

    def _build_model(self, src_ids, position_ids, sentence_ids, task_ids,
                     input_mask):

        emb_out = self._build_embedding(src_ids, position_ids, sentence_ids,
                                        task_ids)
        self.input_mask = input_mask
        self._enc_out, self.all_hidden, self.all_attn, self.all_ffn = encoder(
            enc_input=emb_out,
            input_mask=input_mask,
            n_layer=self._n_layer,
            n_head=self._n_head,
            d_key=self._emb_size // self._n_head,
            d_value=self._emb_size // self._n_head,
            d_model=self._emb_size,
            d_inner_hid=self._emb_size * 4,
            prepostprocess_dropout=self._prepostprocess_dropout,
            attention_dropout=self._attention_dropout,
            relu_dropout=0,
            hidden_act=self._hidden_act,
            preprocess_cmd=self._preprocess_cmd,
            postprocess_cmd=self._postprocess_cmd,
            param_initializer=self._param_initializer,
            name=self.name + 'encoder')
        if self._dtype == "float16":
            self._enc_out = fluid.layers.cast(
                x=self._enc_out, dtype=self._emb_dtype)

    def _build_embedding(self, src_ids, position_ids, sentence_ids, task_ids):
        # padding id in vocabulary must be set to 0
        emb_out = fluid.layers.embedding(
            input=src_ids,
            size=[self._voc_size, self._emb_size],
            dtype=self._emb_dtype,
            param_attr=fluid.ParamAttr(
                name=self._word_emb_name, initializer=self._param_initializer),
            is_sparse=False)

        position_emb_out = fluid.layers.embedding(
            input=position_ids,
            size=[self._max_position_seq_len, self._emb_size],
            dtype=self._emb_dtype,
            param_attr=fluid.ParamAttr(
                name=self._pos_emb_name, initializer=self._param_initializer))

        sent_emb_out = fluid.layers.embedding(
            sentence_ids,
            size=[self._sent_types, self._emb_size],
            dtype=self._emb_dtype,
            param_attr=fluid.ParamAttr(
                name=self._sent_emb_name, initializer=self._param_initializer))

        self.all_emb = [emb_out, position_emb_out, sent_emb_out]
        emb_out = emb_out + position_emb_out
        emb_out = emb_out + sent_emb_out

        if self._use_task_id:
            task_emb_out = fluid.layers.embedding(
                task_ids,
                size=[self._task_types, self._emb_size],
                dtype=self._emb_dtype,
                param_attr=fluid.ParamAttr(
                    name=self._task_emb_name,
                    initializer=self._param_initializer))

            emb_out = emb_out + task_emb_out

        emb_out = pre_process_layer(
            emb_out,
            'nd',
            self._prepostprocess_dropout,
            name=self.name + 'pre_encoder')

        if self._dtype == "float16":
            emb_out = fluid.layers.cast(x=emb_out, dtype=self._dtype)
        return emb_out

    def get_sequence_output(self):
        return self._enc_out

    def get_pooled_output(self):
        """Get the first feature of each sequence for classification"""
        next_sent_feat = self._enc_out[:, 0, :]
        #next_sent_feat = fluid.layers.slice(input=self._enc_out, axes=[1], starts=[0], ends=[1])
        next_sent_feat = fluid.layers.fc(
            input=next_sent_feat,
            size=self._emb_size,
            act="tanh",
            param_attr=fluid.ParamAttr(
                name=self.name + "pooled_fc.w_0",
                initializer=self._param_initializer),
            bias_attr=self.name + "pooled_fc.b_0")
        return next_sent_feat

    def get_lm_output(self, mask_label, mask_pos):
        """Get the loss & accuracy for pretraining"""

        mask_pos = fluid.layers.cast(x=mask_pos, dtype='int32')

        # extract the first token feature in each sentence
        self.next_sent_feat = self.get_pooled_output()
        reshaped_emb_out = fluid.layers.reshape(
            x=self._enc_out, shape=[-1, self._emb_size])
        # extract masked tokens' feature
        mask_feat = fluid.layers.gather(input=reshaped_emb_out, index=mask_pos)

        # transform: fc
        mask_trans_feat = fluid.layers.fc(
            input=mask_feat,
            size=self._emb_size,
            act=self._hidden_act,
            param_attr=fluid.ParamAttr(
                name=self.name + 'mask_lm_trans_fc.w_0',
                initializer=self._param_initializer),
            bias_attr=fluid.ParamAttr(name=self.name + 'mask_lm_trans_fc.b_0'))

        # transform: layer norm 
        mask_trans_feat = fluid.layers.layer_norm(
            mask_trans_feat,
            begin_norm_axis=len(mask_trans_feat.shape) - 1,
            param_attr=fluid.ParamAttr(
                name=self.name + 'mask_lm_trans_layer_norm_scale',
                initializer=fluid.initializer.Constant(1.)),
            bias_attr=fluid.ParamAttr(
                name=self.name + 'mask_lm_trans_layer_norm_bias',
                initializer=fluid.initializer.Constant(0.)))
        # transform: layer norm 
        #mask_trans_feat = pre_process_layer(
        #    mask_trans_feat, 'n', name=self.name + 'mask_lm_trans')

        mask_lm_out_bias_attr = fluid.ParamAttr(
            name=self.name + "mask_lm_out_fc.b_0",
            initializer=fluid.initializer.Constant(value=0.0))
        if self._weight_sharing:
            fc_out = fluid.layers.matmul(
                x=mask_trans_feat,
                y=fluid.default_main_program().global_block().var(
                    self._word_emb_name),
                transpose_y=True)
            fc_out += fluid.layers.create_parameter(
                shape=[self._voc_size],
                dtype=self._emb_dtype,
                attr=mask_lm_out_bias_attr,
                is_bias=True)

        else:
            fc_out = fluid.layers.fc(input=mask_trans_feat,
                                     size=self._voc_size,
                                     param_attr=fluid.ParamAttr(
                                         name=self.name + "mask_lm_out_fc.w_0",
                                         initializer=self._param_initializer),
                                     bias_attr=mask_lm_out_bias_attr)

        mask_lm_loss = fluid.layers.softmax_with_cross_entropy(
            logits=fc_out, label=mask_label)
        return mask_lm_loss

    def get_task_output(self, task, task_labels):
        task_fc_out = fluid.layers.fc(
            input=self.next_sent_feat,
            size=task["num_labels"],
            param_attr=fluid.ParamAttr(
                name=self.name + task["task_name"] + "_fc.w_0",
                initializer=self._param_initializer),
            bias_attr=self.name + task["task_name"] + "_fc.b_0")
        task_loss, task_softmax = fluid.layers.softmax_with_cross_entropy(
            logits=task_fc_out, label=task_labels, return_softmax=True)
        task_acc = fluid.layers.accuracy(input=task_softmax, label=task_labels)
        return task_loss, task_acc


class ErnieGraphModel(ErnieModel):
    def __init__(self,
                 src_ids,
                 task_ids=None,
                 config=None,
                 weight_sharing=True,
                 use_fp16=False,
                 slot_seqlen=40,
                 name=""):
        self.slot_seqlen = slot_seqlen
        self._set_config(config, name, weight_sharing)
        input_mask = self._build_input_mask(src_ids)
        position_ids = self._build_position_ids(src_ids)
        sentence_ids = self._build_sentence_ids(src_ids)
        self._build_model(src_ids, position_ids, sentence_ids, task_ids,
                          input_mask)
        self._debug_summary(input_mask)

    def _build_position_ids(self, src_ids):
        src_shape = L.shape(src_ids)
        src_seqlen = src_shape[1]
        src_batch = src_shape[0]

        slot_seqlen = self.slot_seqlen

        num_b = (src_seqlen / slot_seqlen) - 1
        a_position_ids = L.reshape(
            L.range(
                0, slot_seqlen, 1, dtype='int32'), [1, slot_seqlen, 1],
            inplace=True) # [1, slot_seqlen, 1]
        a_position_ids = L.expand(a_position_ids, [src_batch, 1, 1]) # [B, slot_seqlen * num_b, 1]

        zero = L.fill_constant([1], dtype='int64', value=0)
        input_mask = L.cast(L.equal(src_ids[:, :slot_seqlen], zero), "int32")  # assume pad id == 0 [B, slot_seqlen, 1]
        a_pad_len = L.reduce_sum(input_mask, 1) # [B, 1, 1]

        b_position_ids = L.reshape(
            L.range(
                slot_seqlen, 2*slot_seqlen, 1, dtype='int32'), [1, slot_seqlen, 1],
            inplace=True) # [1, slot_seqlen, 1]
        b_position_ids = L.expand(b_position_ids, [src_batch, num_b, 1]) # [B, slot_seqlen * num_b, 1]
        b_position_ids = b_position_ids - a_pad_len # [B, slot_seqlen * num_b, 1]

        position_ids = L.concat([a_position_ids, b_position_ids], 1)
        position_ids = L.cast(position_ids, 'int64')
        position_ids.stop_gradient = True
        return position_ids

    def _build_sentence_ids(self, src_ids):
        src_shape = L.shape(src_ids)
        src_seqlen = src_shape[1]
        src_batch = src_shape[0]

        slot_seqlen = self.slot_seqlen

        zeros = L.zeros([src_batch, slot_seqlen, 1], "int64")
        ones = L.ones([src_batch, src_seqlen-slot_seqlen, 1], "int64")

        sentence_ids = L.concat([zeros, ones], 1)
        sentence_ids.stop_gradient = True
        return sentence_ids

    def _build_model(self, src_ids, position_ids, sentence_ids, task_ids,
                     input_mask):

        emb_out = self._build_embedding(src_ids, position_ids, sentence_ids,
                                        task_ids)
        self.input_mask = input_mask
        self._enc_out, self.all_hidden, self.all_attn, self.all_ffn = graph_encoder(
            enc_input=emb_out,
            input_mask=input_mask,
            n_layer=self._n_layer,
            n_head=self._n_head,
            d_key=self._emb_size // self._n_head,
            d_value=self._emb_size // self._n_head,
            d_model=self._emb_size,
            d_inner_hid=self._emb_size * 4,
            prepostprocess_dropout=self._prepostprocess_dropout,
            attention_dropout=self._attention_dropout,
            relu_dropout=0,
            hidden_act=self._hidden_act,
            preprocess_cmd=self._preprocess_cmd,
            postprocess_cmd=self._postprocess_cmd,
            param_initializer=self._param_initializer,
            slot_seqlen=self.slot_seqlen,
            name=self.name + 'encoder')
        if self._dtype == "float16":
            self._enc_out = fluid.layers.cast(
                x=self._enc_out, dtype=self._emb_dtype)
