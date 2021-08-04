# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn as nn
from dataloader import BasicDataset
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import pgl
from pgl.nn import functional as GF
from tqdm import tqdm


class BasicModel(nn.Layer):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        raise NotImplementedError


class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()

    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError


class NGCF_Layer(nn.Layer):
    def __init__(self, input_size, output_size, n_layers=3):
        super(NGCF_Layer, self).__init__()
        self.n_layers = n_layers
        self.ngcfs = nn.LayerList()
        for i in range(n_layers):
            self.ngcfs.append(pgl.nn.NGCFConv(input_size, output_size))

    def forward(self, graph, u_feat, i_feat):
        h = paddle.concat([u_feat, i_feat])
        embs = [h]
        for i in range(self.n_layers):
            h = self.ngcfs[i](graph, h)
            norm_h = F.normalize(h, p=2, axis=1)
            embs.append(norm_h)
        embs = paddle.concat(embs, axis=1)
        users, items = paddle.split(embs, [u_feat.shape[0], i_feat.shape[0]])
        return users, items


class NGCF(nn.Layer):
    def __init__(self, args, dataset):
        super(NGCF, self).__init__()
        self.args = args
        self.dataset = dataset
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        num_nodes = self.dataset.n_users + self.dataset.m_items

        self.latent_dim = self.args.recdim
        self.n_layers = self.args.n_layers
        self.ngcf = NGCF_Layer(self.latent_dim, self.latent_dim, self.n_layers)
        edges = paddle.to_tensor(self.dataset.trainEdge, dtype='int64')
        self.Graph = pgl.Graph(num_nodes=num_nodes, edges=edges)
        self.f = nn.Sigmoid()
        self.__init_weight()

    def __init_weight(self):
        weight_attr1 = paddle.framework.ParamAttr(
            initializer=nn.initializer.XavierUniform())
        weight_attr2 = paddle.framework.ParamAttr(
            initializer=nn.initializer.XavierUniform())
        self.embedding_user = nn.Embedding(
            num_embeddings=self.num_users,
            embedding_dim=self.latent_dim,
            weight_attr=weight_attr1)
        self.embedding_item = nn.Embedding(
            num_embeddings=self.num_items,
            embedding_dim=self.latent_dim,
            weight_attr=weight_attr2)

    def getUsersRating(self, users):

        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight

        all_users, all_items = self.ngcf(self.Graph, users_emb, items_emb)

        users_emb = paddle.to_tensor(all_users.numpy()[users.numpy()])
        items_emb = all_items
        rating = self.f(paddle.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_users, all_items = self.ngcf(self.Graph, users_emb, items_emb)
        users_emb = paddle.index_select(all_users, users)
        pos_emb = paddle.index_select(all_items, pos_items)
        neg_emb = paddle.index_select(all_items, neg_items)
        return users_emb, pos_emb, neg_emb

    def bpr_loss(self, users, pos, neg):
        users_emb, pos_emb, neg_emb = self.getEmbedding(
            users.astype('int32'), pos.astype('int32'), neg.astype('int32'))

        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2)
                              + neg_emb.norm(2).pow(2)) / float(len(users))
        pos_scores = paddle.multiply(users_emb, pos_emb)
        pos_scores = paddle.sum(pos_scores, axis=1)
        neg_scores = paddle.multiply(users_emb, neg_emb)
        neg_scores = paddle.sum(neg_scores, axis=1)

        loss = nn.LogSigmoid()(pos_scores - neg_scores)

        loss = -1 * paddle.mean(loss)
        return loss, reg_loss


#     def forward(self, users, items):

#         users_emb = self.embedding_user.weight
#         items_emb = self.embedding_item.weight

#         all_users, all_items = self.lgn(self.Graph, users_emb, items_emb)
#         users_emb = paddle.to_tensor(all_users.numpy()[users.numpy()])
#         items_emb = paddle.to_tensor(all_items.numpy()[items.numpy()])
#         inner_pro = paddle.multiply(users_emb, items_emb)
#         gamma     = paddle.sum(inner_pro, axis=1)
#         return gamma
