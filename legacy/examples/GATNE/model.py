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
"""
This file implement the GATNE model.
"""

import numpy as np
import math
import logging

import paddle.fluid as fluid
import paddle.fluid.layers as fl
from pgl import heter_graph_wrapper


class GATNE(object):
    """Implementation of GATNE model.

    Args:
        config: dict, some configure parameters.
        dataset: instance of Dataset class
        place: GPU or CPU place 
    """

    def __init__(self, config, dataset, place):
        logging.info(['model is: ', self.__class__.__name__])
        self.config = config
        self.graph = dataset.graph
        self.placce = place
        self.edge_types = sorted(self.graph.edge_types_info())
        logging.info('edge_types in model: %s' % str(self.edge_types))
        neg_num = dataset.config['neg_num']

        # hyper parameters
        self.num_nodes = self.graph.num_nodes
        self.embedding_size = self.config['dimensions']
        self.embedding_u_size = self.config['edge_dim']
        self.dim_a = self.config['att_dim']
        self.att_head = self.config['att_head']
        self.edge_type_count = len(self.edge_types)
        self.u_num = self.edge_type_count

        self.gw = heter_graph_wrapper.HeterGraphWrapper(
            name="heter_graph",
            edge_types=self.graph.edge_types_info(),
            node_feat=self.graph.node_feat_info(),
            edge_feat=self.graph.edge_feat_info())

        self.train_inputs = fl.data(
            'train_inputs', shape=[None], dtype='int64')

        self.train_labels = fl.data(
            'train_labels', shape=[None, 1, 1], dtype='int64')

        self.train_types = fl.data(
            'train_types', shape=[None, 1], dtype='int64')

        self.train_negs = fl.data(
            'train_negs', shape=[None, neg_num, 1], dtype='int64')

        self.forward()

    def forward(self):
        """Build the GATNE net.
        """
        param_attr_init = fluid.initializer.Uniform(
            low=-1.0, high=1.0, seed=np.random.randint(100))
        embed_param_attrs = fluid.ParamAttr(
            name='Base_node_embed', initializer=param_attr_init)

        # node_embeddings
        base_node_embed = fl.embedding(
            input=fl.reshape(
                self.train_inputs, shape=[-1, 1]),
            size=[self.num_nodes, self.embedding_size],
            param_attr=embed_param_attrs)

        node_features = []
        for edge_type in self.edge_types:
            param_attr_init = fluid.initializer.Uniform(
                low=-1.0, high=1.0, seed=np.random.randint(100))
            embed_param_attrs = fluid.ParamAttr(
                name='%s_node_embed' % edge_type, initializer=param_attr_init)

            features = fl.embedding(
                input=self.gw[edge_type].node_feat['index'],
                size=[self.num_nodes, self.embedding_u_size],
                param_attr=embed_param_attrs)

            node_features.append(features)

        # mp_output: list of embedding(self.num_nodes, dim)
        mp_output = self.message_passing(self.gw, self.edge_types,
                                         node_features)

        # U : (num_type[m], num_nodes, dim[s])
        node_type_embed = fl.stack(mp_output, axis=0)

        # U : (num_nodes, num_type[m], dim[s])
        node_type_embed = fl.transpose(node_type_embed, perm=[1, 0, 2])

        #gather node_type_embed from train_inputs
        node_type_embed = fl.gather(node_type_embed, self.train_inputs)

        # M_r
        tn_initializer = fluid.initializer.TruncatedNormalInitializer(
            loc=0.0, scale=1.0 / math.sqrt(self.embedding_size))

        trans_weights = fl.create_parameter(
            shape=[
                self.edge_type_count, self.embedding_u_size,
                self.embedding_size // self.att_head
            ],
            default_initializer=tn_initializer,
            dtype='float32',
            name='trans_w')

        # W_r
        trans_weights_s1 = fl.create_parameter(
            shape=[self.edge_type_count, self.embedding_u_size, self.dim_a],
            default_initializer=tn_initializer,
            dtype='float32',
            name='trans_w_s1')

        # w_r
        trans_weights_s2 = fl.create_parameter(
            shape=[self.edge_type_count, self.dim_a, self.att_head],
            default_initializer=tn_initializer,
            dtype='float32',
            name='trans_w_s2')

        trans_w = fl.gather(trans_weights, self.train_types)
        trans_w_s1 = fl.gather(trans_weights_s1, self.train_types)
        trans_w_s2 = fl.gather(trans_weights_s2, self.train_types)

        attention = self.attention(node_type_embed, trans_w_s1, trans_w_s2)
        node_type_embed = fl.matmul(attention, node_type_embed)
        node_embed = base_node_embed + fl.reshape(
            fl.matmul(node_type_embed, trans_w), [-1, self.embedding_size])

        self.last_node_embed = fl.l2_normalize(node_embed, axis=1)

        nce_weight_initializer = fluid.initializer.TruncatedNormalInitializer(
            loc=0.0, scale=1.0 / math.sqrt(self.embedding_size))
        nce_weight_attrs = fluid.ParamAttr(
            name='nce_weight', initializer=nce_weight_initializer)

        weight_pos = fl.embedding(
            input=self.train_labels,
            size=[self.num_nodes, self.embedding_size],
            param_attr=nce_weight_attrs)
        weight_neg = fl.embedding(
            input=self.train_negs,
            size=[self.num_nodes, self.embedding_size],
            param_attr=nce_weight_attrs)
        tmp_node_embed = fl.unsqueeze(self.last_node_embed, axes=[1])
        pos_logits = fl.matmul(
            tmp_node_embed, weight_pos, transpose_y=True)  # [B, 1, 1]

        neg_logits = fl.matmul(
            tmp_node_embed, weight_neg, transpose_y=True)  # [B, 1, neg_num]

        pos_score = fl.squeeze(pos_logits, axes=[1])
        pos_score = fl.clip(pos_score, min=-10, max=10)
        pos_score = -1.0 * fl.logsigmoid(pos_score)

        neg_score = fl.squeeze(neg_logits, axes=[1])
        neg_score = fl.clip(neg_score, min=-10, max=10)
        neg_score = -1.0 * fl.logsigmoid(-1.0 * neg_score)

        neg_score = fl.reduce_sum(neg_score, dim=1, keep_dim=True)
        self.loss = fl.reduce_mean(pos_score + neg_score)

    def attention(self, node_type_embed, trans_w_s1, trans_w_s2):
        """Calculate attention weights.
        """
        attention = fl.tanh(fl.matmul(node_type_embed, trans_w_s1))
        attention = fl.matmul(attention, trans_w_s2)
        attention = fl.reshape(attention, [-1, self.u_num])
        attention = fl.softmax(attention)
        attention = fl.reshape(attention, [-1, self.att_head, self.u_num])
        return attention

    def message_passing(self, gw, edge_types, features, name=''):
        """Message passing from source nodes to dstination nodes
        """

        def __message(src_feat, dst_feat, edge_feat):
            """send function
            """
            return src_feat['h']

        def __reduce(feat):
            """recv function
            """
            return fluid.layers.sequence_pool(feat, pool_type='average')

        if not isinstance(edge_types, list):
            edge_types = [edge_types]

        if not isinstance(features, list):
            features = [features]

        assert len(edge_types) == len(features)

        output = []
        for i in range(len(edge_types)):
            msg = gw[edge_types[i]].send(
                __message, nfeat_list=[('h', features[i])])
            neigh_feat = gw[edge_types[i]].recv(msg, __reduce)
            neigh_feat = fl.fc(neigh_feat,
                               size=neigh_feat.shape[-1],
                               name='neigh_fc_%d' % (i),
                               act='sigmoid')
            slf_feat = fl.fc(features[i],
                             size=neigh_feat.shape[-1],
                             name='slf_fc_%d' % (i),
                             act='sigmoid')

            out = fluid.layers.concat([slf_feat, neigh_feat], axis=1)
            out = fl.fc(out, size=neigh_feat.shape[-1], name='fc', act=None)
            out = fluid.layers.l2_normalize(out, axis=1)
            output.append(out)

        # list of matrix
        return output

    def backward(self, global_steps, opt_config):
        """Build the optimizer.
        """
        self.lr = fl.polynomial_decay(opt_config['lr'], global_steps, 0.001)
        adam = fluid.optimizer.Adam(learning_rate=self.lr)
        adam.minimize(self.loss)
