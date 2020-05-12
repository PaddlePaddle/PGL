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
import time
import glob
import os

import numpy as np

import pgl
import paddle.fluid as F
import paddle.fluid.layers as L

from models import message_passing

def get_layer(layer_type, gw, feature, hidden_size, act, initializer, learning_rate, name, is_test=False):
    return getattr(message_passing, layer_type)(gw, feature, hidden_size, act, initializer, learning_rate, name)


class BaseGraphWrapperBuilder(object):
    def __init__(self, config):
        self.config = config
        self.node_feature_info = []
        self.edge_feature_info = []

    def __call__(self):
        place = F.CPUPlace()
        graph_wrappers = []
        for i in range(self.config.num_layers):
            # all graph have same node_feat_info
            graph_wrappers.append(
                pgl.graph_wrapper.GraphWrapper(
                    "layer_%s" % i, node_feat=self.node_feature_info, edge_feat=self.edge_feature_info))
        return graph_wrappers


class GraphsageGraphWrapperBuilder(BaseGraphWrapperBuilder):
    def __init__(self, config):
        super(GraphsageGraphWrapperBuilder, self).__init__(config)
        self.node_feature_info.append(('index', [None], np.dtype('int64')))


class BaseGNNModel(object):
    def __init__(self, config):
        self.config = config
        self.graph_wrapper_builder = self.gen_graph_wrapper_builder(config) 
        self.net_fn = self.gen_net_fn(config)
        self.feed_list_builder = self.gen_feed_list_builder(config)
        self.data_loader_builder = self.gen_data_loader_builder(config)
        self.loss_fn = self.gen_loss_fn(config)
        self.build()


    def gen_graph_wrapper_builder(self, config): 
        return GraphsageGraphWrapperBuilder(config)

    def gen_net_fn(self, config):
        return BaseNet(config)

    def gen_feed_list_builder(self, config):
        return BaseFeedListBuilder(config) 

    def gen_data_loader_builder(self, config):
        return BaseDataLoaderBuilder(config)

    def gen_loss_fn(self, config):
        return BaseLoss(config)

    def build(self):
        self.graph_wrappers = self.graph_wrapper_builder()
        self.inputs, self.outputs = self.net_fn(self.graph_wrappers)
        self.feed_list = self.feed_list_builder(self.inputs, self.graph_wrappers)
        self.data_loader = self.data_loader_builder(self.feed_list)
        self.loss = self.loss_fn(self.outputs)

class BaseFeedListBuilder(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, inputs, graph_wrappers):
        feed_list = []
        for i in range(len(graph_wrappers)):
            feed_list.extend(graph_wrappers[i].holder_list)
        feed_list.extend(inputs)
        return feed_list


class BaseDataLoaderBuilder(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, feed_list):
        data_loader = F.io.PyReader(
            feed_list=feed_list, capacity=20, use_double_buffer=True, iterable=True)
        return data_loader



class BaseNet(object):
    def __init__(self, config):
        self.config = config

    def take_final_feature(self, feature, index, name):
        """take final feature"""
        feat = L.gather(feature, index, overwrite=False)

        if self.config.final_fc:
            feat = L.fc(feat,
                           self.config.hidden_size,
                           param_attr=F.ParamAttr(name=name + '_w'),
                           bias_attr=F.ParamAttr(name=name + '_b'))

        if self.config.final_l2_norm:
            feat = L.l2_normalize(feat, axis=1)
        return feat

    def build_inputs(self):
        user_index = L.data(
            "user_index", shape=[None], dtype="int64", append_batch_size=False)
        item_index = L.data(
            "item_index", shape=[None], dtype="int64", append_batch_size=False)
        neg_item_index = L.data(
            "neg_item_index", shape=[None], dtype="int64", append_batch_size=False)
        return [user_index, item_index, neg_item_index]

    def build_embedding(self, graph_wrappers, inputs=None):
        num_embed = int(np.load(os.path.join(self.config.graph_path, "num_nodes.npy")))
        is_sparse = self.config.trainer_type == "Transpiler"

        embed = L.embedding(
            input=L.reshape(graph_wrappers[0].node_feat['index'], [-1, 1]),
            size=[num_embed, self.config.hidden_size],
            is_sparse=is_sparse,
            param_attr=F.ParamAttr(name="node_embedding", initializer=F.initializer.Uniform(
                low=-1. / self.config.hidden_size,
                high=1. / self.config.hidden_size)))
        return embed

    def gnn_layers(self, graph_wrappers, feature):
        features = [feature]

        initializer = None
        fc_lr = self.config.lr / 0.001

        for i in range(self.config.num_layers):
            if i == self.config.num_layers - 1:
                act = None
            else:
                act = "leaky_relu"

            feature = get_layer(
                self.config.layer_type,
                graph_wrappers[i],
                feature,
                self.config.hidden_size,
                act,
                initializer,
                learning_rate=fc_lr,
                name="%s_%s" % (self.config.layer_type, i))
            features.append(feature)
        return features

    def __call__(self, graph_wrappers):
        inputs = self.build_inputs()
        feature = self.build_embedding(graph_wrappers, inputs)
        features = self.gnn_layers(graph_wrappers, feature)
        outputs = [self.take_final_feature(features[-1], i, "final_fc") for i in inputs]
        src_real_index = L.gather(graph_wrappers[0].node_feat['index'], inputs[0])
        outputs.append(src_real_index)
        return inputs, outputs

def all_gather(X):
    trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
    trainer_num = int(os.getenv("PADDLE_TRAINERS_NUM", "0"))
    if trainer_num == 1:
        copy_X = X * 1
        copy_X.stop_gradients=True
        return copy_X

    Xs = []
    for i in range(trainer_num):
        copy_X = X * 1
        copy_X =  L.collective._broadcast(copy_X, i, True)
        copy_X.stop_gradients=True
        Xs.append(copy_X)

    if len(Xs) > 1:
        Xs=L.concat(Xs, 0)
        Xs.stop_gradients=True
    else:
        Xs = Xs[0]
    return Xs

class BaseLoss(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, outputs):
        user_feat, item_feat, neg_item_feat = outputs[0], outputs[1], outputs[2]
        loss_type = self.config.loss_type

        if self.config.neg_type == "batch_neg":
            neg_item_feat = item_feat
        # Calc Loss
        if self.config.loss_type == "hinge":
            pos = L.reduce_sum(user_feat * item_feat, -1, keep_dim=True) # [B, 1]
            neg = L.matmul(user_feat, neg_item_feat, transpose_y=True) # [B, B]
            loss = L.reduce_mean(L.relu(neg - pos + self.config.margin))
        elif self.config.loss_type == "all_hinge":
            pos = L.reduce_sum(user_feat * item_feat, -1, keep_dim=True) # [B, 1]
            all_pos = all_gather(pos) # [B * n, 1]
            all_neg_item_feat = all_gather(neg_item_feat) # [B * n, 1]
            all_user_feat = all_gather(user_feat) # [B * n, 1]

            neg1 = L.matmul(user_feat, all_neg_item_feat, transpose_y=True) # [B, B * n]
            neg2 = L.matmul(all_user_feat, neg_item_feat, transpose_y=True) # [B *n, B]

            loss1 = L.reduce_mean(L.relu(neg1 - pos + self.config.margin))
            loss2 = L.reduce_mean(L.relu(neg2 - all_pos + self.config.margin))

            #loss = (loss1 + loss2) / 2
            loss = loss1 + loss2
        
        elif self.config.loss_type == "softmax":
            pass
            # TODO
            # pos = L.reduce_sum(user_feat * item_feat, -1, keep_dim=True) # [B, 1]
            # neg = L.matmul(user_feat, neg_feat, transpose_y=True) # [B, B]
            # logits = L.concat([pos, neg], -1) # [B, 1+B]
            # labels = L.fill_constant_batch_size_like(logits, [-1, 1], "int64", 0)
            # loss = L.reduce_mean(L.softmax_with_cross_entropy(logits, labels))
        else:
            raise ValueError
        return loss
