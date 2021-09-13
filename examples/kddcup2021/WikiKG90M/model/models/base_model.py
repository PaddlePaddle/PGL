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
from .base_loss import LossFunc


def uniform(low, high, size, dtype=np.float32):
    rng = np.random.default_rng()
    out = (high - low) * rng.random(size, dtype=dtype) + low
    return out


class NumpyEmbedding():
    def __init__(self,
                 num_embedding,
                 embedding_dim,
                 init_value=None,
                 relation_times=1,
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
                weight.reshape((-1, relation_times + 1
                                ))[:, -1] = 1.0 if scale_type == 1 else 0.0
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
                    relation_times=1,
                    init_value=None,
                    is_cpu=False,
                    scale_type=0,
                    name=None,
                    weight_path=None):

    use_scale = True if scale_type > 0 else False
    if is_cpu:
        embedding = NumpyEmbedding(num_embedding, embedding_dim, init_value,
                                   relation_times, scale_type, weight_path)
        print("in cpu embedding.")
        return embedding
    embedding = nn.Embedding(
        num_embedding, embedding_dim, weight_attr=paddle.ParamAttr(name=name))
    if init_value is not None:
        if use_scale:
            init_value = 1. / math.sqrt(embedding_dim)
            value = uniform(-init_value, init_value,
                            [num_embedding, embedding_dim])
            scale_value = 1.0 if scale_type == 1 else 0.0
            value.reshape((-1, relation_times + 1))[:, -1] = scale_value

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


class BaseKEModel(nn.Layer):
    """
    Base knowledge graph embedding model.
    """

    def __init__(self,
                 args,
                 n_entities,
                 n_relations,
                 model_name,
                 hidden_size,
                 entity_feat_dim,
                 relation_feat_dim,
                 gamma=None,
                 double_entity_emb=False,
                 relation_times=1,
                 scale_type=0):
        super(BaseKEModel, self).__init__()
        self.args = args
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.model_name = model_name
        self.hidden_size = hidden_size
        self.entity_feat_dim = entity_feat_dim
        self.relation_feat_dim = relation_feat_dim
        self.relation_times = relation_times

        # TODO: Optional parameters
        self.gamma = gamma

        self.eps = 2.0
        self.use_scale = True if scale_type > 0 else False

        self.entity_dim = 2 * self.hidden_size if double_entity_emb else self.hidden_size
        self.relation_dim = hidden_size * (
            self.relation_times + int(self.use_scale))
        self.emb_init = (gamma + self.eps) / self.hidden_size

        self.entity_embedding = embedding_layer(
            self.n_entities,
            self.entity_dim,
            init_value=self.emb_init,
            is_cpu=args.cpu_emb,
            name="entity_embedding",
            weight_path=args.weight_path)
        self.entity_feat = None

        self.relation_embedding = embedding_layer(
            self.n_relations,
            self.relation_dim,
            init_value=self.emb_init,
            is_cpu=False,
            relation_times=relation_times,
            scale_type=scale_type,
            name="relation_embedding")
        self.relation_feat = None

        if self.args.use_feature:
            self.transform_net = MLP(
                self.entity_dim + entity_feat_dim, self.entity_dim,
                self.relation_dim + relation_feat_dim, self.relation_dim)

        if self.model_name == "TransE":
            self.score_function = TransEScore(self.gamma)
        elif self.model_name == "RotatE":
            self.score_function = RotatEScore(self.gamma, self.emb_init)
        elif self.model_name == "OTE":
            self.score_function = OTEScore(self.gamma, relation_times,
                                           scale_type)
            if self.use_scale:
                scale_value = 1.0 if scale_type == 1 else 0.0
                self.relation_embedding.weight.reshape(
                    [-1, relation_times + 1])[:, -1] = scale_value
            orth_embedding = self.score_function.orth_rel_embedding(
                self.relation_embedding.weight)
            self.relation_embedding.weight.set_value(orth_embedding)

        self.loss_func = LossFunc(
            args,
            loss_type=args.loss_genre,
            neg_adv_sampling=args.neg_adversarial_sampling,
            adv_temp_value=args.adversarial_temperature,
            pairwise=args.pairwise)

    def forward(self, pos_triplets, neg_ents, real_ent_ids,
                neg_head_mode=True):
        entity_emb = self.entity_embedding(real_ent_ids)
        if not self.args.cpu_emb:
            self.entity_embedding.curr_emb = entity_emb
        if self.args.use_feature:
            entity_feat = paddle.to_tensor(self.entity_feat[real_ent_ids.numpy(
            )].astype('float32'))
            emb = paddle.concat([entity_feat, entity_emb], axis=-1)

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
            pos_head = F.embedding(pos_triplets[0], entity_emb)
            pos_tail = F.embedding(pos_triplets[2], entity_emb)
            neg_ents = F.embedding(neg_ents, entity_emb)
            pos_rel = self.relation_embedding(pos_triplets[1])

        batch_size = pos_head.shape[0]
        if batch_size < self.args.neg_sample_size:
            neg_sample_size = batch_size
        else:
            neg_sample_size = self.args.neg_sample_size
        pos_score = self.score_function.get_score(pos_head, pos_rel, pos_tail)
        if neg_head_mode:
            neg_score = self.score_function.get_neg_score(
                neg_ents, pos_rel, pos_tail, batch_size, neg_sample_size,
                neg_sample_size, neg_head_mode)
        else:
            neg_score = self.score_function.get_neg_score(
                pos_head, pos_rel, neg_ents, batch_size, neg_sample_size,
                neg_sample_size, neg_head_mode)
        loss = self.loss_func.get_total_loss(pos_score, neg_score)
        return loss

    def forward_test_wikikg(self, query, ans, candidate, mode='h,r->t'):
        scores = self.predict_wikikg_score(query, candidate, mode)
        argsort = paddle.argsort(scores, axis=1, descending=True)
        return argsort[:, :10], scores

    def predict_wikikg_score(self, query, candidate, mode):
        if mode == 'h,r->t':
            neg_sample_size = candidate.shape[1]
            if self.args.use_feature:
                neg_tail_entity_emb = self.entity_embedding(
                    paddle.reshape(candidate, [-1]))
                neg_tail_entity_feat = paddle.to_tensor(self.entity_feat[
                    paddle.reshape(candidate, [-1]).numpy()].astype('float32'))
                neg_tail = self.transform_net.embed_entity(
                    paddle.concat([neg_tail_entity_feat, neg_tail_entity_emb],
                                  -1))

                head_emb = self.entity_embedding(query[:, 0])
                head_feat = paddle.to_tensor(self.entity_feat[
                    query[:, 0].numpy()].astype('float32'))
                head = self.transform_net.embed_entity(
                    paddle.concat([head_feat, head_emb], -1))

                rel = self.transform_net.embed_relation(
                    paddle.concat(
                        [
                            paddle.to_tensor(self.relation_feat[
                                query[:, 1].numpy()].astype('float32')),
                            self.relation_embedding(query[:, 1]),
                        ],
                        axis=-1))
            else:
                head = self.entity_embedding(query[:, 0])
                neg_tail = self.entity_embedding(
                    paddle.reshape(candidate, [-1]))
                rel = self.relation_embedding(query[:, 1])
            neg_score = self.score_function.get_neg_score(
                head,
                rel,
                neg_tail,
                batch_size=query.shape[0],
                mini_batch_size=1,
                neg_sample_size=neg_sample_size,
                neg_head=False)
        else:
            assert False
        return paddle.squeeze(neg_score, axis=1)

    def predict(self, test_triplet):
        test_triplet = paddle.to_tensor(test_triplet[0])
        head_vec = self.entity_embedding(test_triplet[0])
        rel_vec = self.relation_embedding(test_triplet[1])
        tail_vec = self.entity_embedding(test_triplet[2])

        head_score, tail_score = self.score_function.get_test_score(
            self.entity_embedding.weight, head_vec, rel_vec, tail_vec)
        return head_score, tail_score

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
