# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved
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
Base model of the knowledge graph embedding models.
"""
import os
import math
import time
import json

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

# import ipdb
from .score_functions import TransEScore, RotatEScore, OTEScore
from utils.helper import timer_wrapper


def uniform(low, high, size, dtype=np.float32):
    rng = np.random.default_rng()
    out = (high - low) * rng.random(size, dtype=dtype) + low
    return out


class NumpyEmbedding():
    def __init__(self,
                 num_embedding,
                 embedding_dim,
                 init_value=None,
                 rel_times=1,
                 scale_type=0,
                 weight_path=None):
        self.weight_path = weight_path

        self.weight = None
        self.tensor_name_dict = []
        self.index_name_dict = []

        self.__load_mmap()
        self.unique_name_count = 0
        self.moment = None
        # deafult optimizer
        self.optim_mode = 'adagrad'
        self.optim_dict = {
            'adagrad': self.update_adagrad,
            'sgd': self.update_sgd
        }
        self.update = self.optim_dict[self.optim_mode]
        self.unique_name_count = 0
        self._init_optimizer()

    def _init_optimizer(self):
        if self.optim_mode in ['adagrad']:
            self.moment = np.zeros_like(self.weight, dtype=np.float32)
            self.moment_path = '%s_moment_%d.npy' % (
                self.weight_path.split('.')[0], time.time() % 1e4)
            np.save(self.moment_path, self.moment)
            del self.moment
            self.moment = np.load(self.moment_path, mmap_mode='r+')

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, index):
        self.index_name_dict = []
        self.tensor_name_dict = []
        if type(index) is paddle.Tensor:
            index = index.numpy()
        # assert len(self.index_name_dict) == 0, "Error, you can only call forward step once in program."
        feat = self.weight[index]
        feat_tensor = paddle.to_tensor(feat)
        feat_tensor.stop_gradient = False
        self.tensor_name_dict.append(feat_tensor)
        self.index_name_dict.append(index)
        return feat_tensor

    @property
    def curr_emb(self):
        return self.tensor_name_dict[0]

    def get(self, index):
        assert type(index) is numpy.ndarray
        return self.weight[index]

    def step(self, lr=None):
        grad_tensor = self.tensor_name_dict[0].grad
        index = self.index_name_dict[0]
        self.update(grad_tensor, index, lr)
        self.index_name_dict = []
        self.tensor_name_dict = []

    def update_adagrad(self, grad, index, lr):
        if self.moment is None:
            self.moment = np.zeros_like(self.weight, dtype=np.float32)
        assert type(grad) is paddle.Tensor
        grad_square = grad * grad
        self.moment[index] += grad_square.numpy()
        std = paddle.to_tensor(self.moment[index])
        std_values = std.sqrt(
        ) + 1e-10  # default paddle 1e-6, 1e-10 for torch.
        update = -lr * grad / std_values
        self.weight[index] += update.numpy()

    def update_sgd(self, grad, lr):
        update = -lr * grad
        self.weight[v] += update.numpy()

    @classmethod
    def create_emb(cls, num_embedding, embedding_dim, init_value, scale_type,
                   weight_path):
        use_scale = True if scale_type > 0 else False
        if init_value:
            if use_scale:
                init_value = 1. / math.sqrt(embedding_dim)
                weight = uniform(-init_value, init_value,
                                 [num_embedding, embedding_dim])
                weight.reshape(
                    (-1,
                     rel_times + 1))[:, -1] = 1.0 if scale_type == 1 else 0.0
                weight.reshape([num_embedding, embedding_dim])

            else:
                weight = uniform(-init_value, init_value,
                                 [num_embedding, embedding_dim])
        np.save(weight_path, weight)
        del weight

    def __load_mmap(self):
        self.weight = np.load(self.weight_path, mmap_mode='r+')


def embedding_layer(num_embedding,
                    embedding_dim,
                    rel_times=1,
                    init_value=None,
                    is_cpu=False,
                    scale_type=0,
                    name=None,
                    weight_path=None):

    use_scale = True if scale_type > 0 else False
    if is_cpu:
        embedding = NumpyEmbedding(num_embedding, embedding_dim, init_value,
                                   rel_times, scale_type, weight_path)
        return embedding
    embedding = nn.Embedding(
        num_embedding, embedding_dim, weight_attr=paddle.ParamAttr(name=name))
    if init_value is not None:
        if use_scale:
            init_value = 1. / math.sqrt(embedding_dim)
            value = uniform(-init_value, init_value,
                            [num_embedding, embedding_dim])
            scale_value = 1.0 if scale_type == 1 else 0.0
            value.reshape((-1, rel_times + 1))[:, -1] = scale_value

        else:
            value = uniform(-init_value, init_value,
                            [num_embedding, embedding_dim])

        embedding.weight.set_value(value)

    return embedding


class MLP(nn.Layer):
    def __init__(self, input_entity_dim, entity_dim, input_relation_dim,
                 relation_dim):
        super(MLP, self).__init__()
        self.transform_e_net = nn.Linear(
            input_entity_dim,
            entity_dim,
            weight_attr=nn.initializer.XavierUniform())
        self.transform_r_net = nn.Linear(
            input_relation_dim,
            relation_dim,
            weight_attr=nn.initializer.XavierUniform())

        a = np.sqrt(6. / (input_entity_dim + entity_dim))
        self.transform_e_net.weight.set_value(
            uniform(
                low=-a, high=a, size=[input_entity_dim, entity_dim]))
        a = np.sqrt(6. / (input_relation_dim + relation_dim))
        self.transform_r_net.weight.set_value(
            uniform(
                low=-a, high=a, size=[input_relation_dim, relation_dim]))

        self.reset_parameters()

    def embed_entity(self, embeddings):
        return self.transform_e_net(embeddings)

    def embed_relation(self, embeddings):
        return self.transform_r_net(embeddings)

    def reset_parameters(self):
        pass


class KGEModel(nn.Layer):
    """
    Base knowledge graph embedding model.
    """

    @timer_wrapper('model construction')
    def __init__(self,
                 args,
                 num_ents,
                 num_rels,
                 score,
                 embed_dim,
                 ent_feat,
                 rel_feat,
                 gamma=None,
                 ent_times=False,
                 rel_times=1,
                 scale_type=0):
        super(KGEModel, self).__init__()
        self.args = args
        self.num_ents = num_ents
        self.num_rels = num_rels
        self.score = score
        self.embed_dim = embed_dim
        self.ent_feat = ent_feat
        self.rel_feat = rel_feat
        self.rel_times = rel_times

        # TODO: Optional parameters
        self.gamma = gamma

        self.eps = 2.0
        self.use_scale = True if scale_type > 0 else False

        self.entity_dim = 2 * self.embed_dim if ent_times else self.embed_dim
        self.relation_dim = embed_dim * (self.rel_times + int(self.use_scale))
        self.emb_init = (gamma + self.eps) / self.embed_dim

        self.entity_embedding = embedding_layer(
            self.num_ents,
            self.entity_dim,
            init_value=self.emb_init,
            is_cpu=args.cpu_emb,
            name="entity_embedding",
            weight_path=args.embs_path)
        self.entity_feat = None

        self.relation_embedding = embedding_layer(
            self.num_rels,
            self.relation_dim,
            init_value=self.emb_init,
            is_cpu=False,
            rel_times=rel_times,
            scale_type=scale_type,
            name="relation_embedding")
        self.relation_feat = None

        if self.args.use_feature:
            self.transform_net = MLP(
                self.entity_dim + ent_feat, self.entity_dim,
                self.relation_dim + rel_feat, self.relation_dim)

        if self.score == "TransE":
            self.score_function = TransEScore(self.gamma)
        elif self.score == "RotatE":
            self.score_function = RotatEScore(self.gamma, self.emb_init)
        elif self.score == "OTE":
            self.score_function = OTEScore(self.gamma, rel_times, scale_type)
            if self.use_scale:
                scale_value = 1.0 if scale_type == 1 else 0.0
                self.relation_embedding.weight.reshape(
                    [-1, rel_times + 1])[:, -1] = scale_value
            orth_embedding = self.score_function.orth_rel_embedding(
                self.relation_embedding.weight)
            self.relation_embedding.weight.set_value(orth_embedding)

    def forward(self, pos_triplets, neg_ents, all_ids, neg_head_mode=True):
        ent_emb = self.entity_embedding(all_ids)
        if not self.args.cpu_emb:
            self.entity_embedding.curr_emb = ent_emb
        if self.args.use_feature:
            ent_feat = paddle.to_tensor(self.entity_feat[all_ids.numpy()]
                                        .astype('float32'))
            emb = paddle.concat([ent_feat, ent_emb], axis=-1)

            pos_head = self.transform_net.embed_entity(
                F.embedding(pos_triplets[0], emb))
            pos_tail = self.transform_net.embed_entity(
                F.embedding(pos_triplets[2], emb))

            neg_ents = self.transform_net.embed_entity(
                F.embedding(neg_ents, emb))

            pos_rel = self.transform_net.embed_relation(
                paddle.concat(
                    [
                        paddle.to_tensor(self.relation_feat[pos_triplets[
                            1].numpy()].astype('float32')),
                        self.relation_embedding(pos_triplets[1])
                    ],
                    axis=-1))
        else:
            pos_head = F.embedding(pos_triplets[0], ent_emb)
            pos_tail = F.embedding(pos_triplets[2], ent_emb)
            neg_ents = F.embedding(neg_ents, ent_emb)
            pos_rel = self.relation_embedding(pos_triplets[1])

        pos_head = paddle.unsqueeze(pos_head, axis=1)
        pos_tail = paddle.unsqueeze(pos_tail, axis=1)
        pos_rel = paddle.unsqueeze(pos_rel, axis=1)

        batch_size = pos_head.shape[0]
        if batch_size < self.args.num_negs:
            neg_sample_size = batch_size
        else:
            neg_sample_size = self.args.num_negs
        pos_score = self.score_function.get_score(pos_head, pos_rel, pos_tail)
        if neg_head_mode:
            neg_score = self.score_function.get_neg_score(
                neg_ents, pos_rel, pos_tail, batch_size, neg_sample_size,
                neg_sample_size, neg_head_mode)
        else:
            neg_score = self.score_function.get_neg_score(
                pos_head, pos_rel, neg_ents, batch_size, neg_sample_size,
                neg_sample_size, neg_head_mode)
        return {'pos': pos_score, 'neg': neg_score}

    def predict(self, ent, rel, cand, mode='tail'):
        num_cands = cand.shape[1]
        if self.args.use_feature is True:

            def mlp_emb(index, embs, feats, trans_net):
                index = paddle.reshape(index, (-1, ))
                emb = embs(index)
                feat = paddle.to_tensor(feats[index.numpy()].astype('float32'))
                emb = trans_net(paddle.concat([feat, emb], axis=-1))
                return emb

            ent = mlp_emb(ent, self.entity_embedding, self.entity_feat,
                          self.transform_net.embed_entity)
            rel = mlp_emb(rel, self.relation_embedding, self.relation_feat,
                          self.transform_net.embed_relation)
            cand = mlp_emb(cand, self.entity_embedding, self.entity_feat,
                           self.transform_net.embed_entity)
        else:
            ent = self.entity_embedding(ent)
            rel = self.relation_embedding(rel)
            cand = self.entity_embedding(paddle.reshape(cand, (-1, )))

        ent = paddle.unsqueeze(ent, axis=1)
        rel = paddle.unsqueeze(rel, axis=1)
        cand = paddle.reshape(cand, (1, num_cands, -1))

        if mode == 'head':
            scores = self.score_function.get_neg_score(
                cand,
                rel,
                ent,
                batch_size=ent.shape[0],
                mini_batch_size=1,
                neg_sample_size=num_cands,
                neg_head=True)
        else:
            scores = self.score_function.get_neg_score(
                ent,
                rel,
                cand,
                batch_size=ent.shape[0],
                mini_batch_size=1,
                neg_sample_size=num_cands,
                neg_head=False)

        scores = paddle.squeeze(scores, axis=1)
        return scores

    def save_model(self):
        """
        Save the model for the predict.
        """
        if not os.path.exists(self.args.save_path):
            os.mkdir(args.save_path)
        conf_file = os.path.join(self.args.save_path, 'model_config.json')
        dict = {}
        config = self.args
        dict.update(vars(config))
        with open(conf_file, 'w') as outfile:
            json.dump(dict, outfile, indent=4)

        # Save the MLP parameters
        paddle.save(self.transform_net.state_dict(),
                    os.path.join(self.args.save_path, "mlp.pdparams"))
        paddle.save(self.relation_embedding.state_dict(),
                    os.path.join(self.args.save_path, "relation.pdparams"))

        # Save the entity and relation embedding
        np.save(
            os.path.join(self.args.save_path, "entity_params"),
            self.entity_embedding.weight)
