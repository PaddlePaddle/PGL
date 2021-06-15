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
import json
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

# import ipdb
from .score_functions import *
from .base_loss import LossFunc


class NumpyEmbedding():
    def __init__(self,
                 num_embedding,
                 embedding_dim,
                 init_value=None,
                 relation_times=1,
                 scale_type=0):
        self.weight = None
        self.tensor_name_dict = []
        self.index_name_dict = []

        use_scale = True if scale_type > 0 else False
        if init_value:
            if use_scale:
                init_value = 1. / math.sqrt(self.embedding_dim)
                self.weight = np.random.uniform(
                    -init_value, init_value,
                    [num_embedding, embedding_dim]).astype(np.float32)
                self.weight.reshape(
                    (-1, relation_times + 1
                     ))[:, -1] = 1.0 if scale_type == 1 else 0.0
                self.weight.reshape([num_embedding, embedding_dim])

            else:
                self.weight = np.random.uniform(
                    -init_value, init_value,
                    [num_embedding, embedding_dim]).astype(np.float32)
        self.unique_name_count = 0
        self.moment = None
        # deafult optimizer
        self.update = self.update_adagrad

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, index):
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
        std_values = std.sqrt() + 1e-6
        update = -lr * grad / std_values
        self.weight[index] += update.numpy()

    def update_sgd(self, grad, lr):
        update = -lr * grad
        self.weight[v] += update.numpy()


def embedding_layer(num_embedding,
                    embedding_dim,
                    relation_times=1,
                    init_value=None,
                    is_cpu=False,
                    scale_type=0,
                    name=None):

    use_scale = True if scale_type > 0 else False
    if is_cpu:
        embedding = NumpyEmbedding(num_embedding, embedding_dim, init_value,
                                   relation_times, scale_type)
        print("in cpu embedding.")
        return embedding
    embedding = nn.Embedding(
        num_embedding, embedding_dim, weight_attr=paddle.ParamAttr(name=name))
    if init_value is not None:
        # init1 = paddle.uniform(
        #     shape=[num_embedding, embedding_dim],
        #     min=-init_value,
        #     max=init_value)
        if use_scale:
            init_value = 1. / math.sqrt(self.embedding_di)
            value = np.random.uniform(
                -init_value, init_value,
                [num_embedding, embedding_dim]).astype(np.float32)
            value.reshape(
                (-1,
                 relation_times + 1))[:, -1] = 1.0 if scale_type == 1 else 0.0

        else:
            value = np.random.uniform(
                -init_value, init_value,
                [num_embedding, embedding_dim]).astype(np.float32)

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
            np.random.uniform(
                low=-a, high=a, size=[input_entity_dim, entity_dim]).astype(
                    np.float32))
        a = np.sqrt(6. / (input_relation_dim + relation_dim))
        self.transform_r_net.weight.set_value(
            np.random.uniform(
                low=-a, high=a, size=[input_relation_dim, relation_dim])
            .astype(np.float32))

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
            name="entity_embedding")
        self.entity_feat = None

        self.relation_embedding = embedding_layer(
            self.n_relations,
            self.relation_dim,
            init_value=self.emb_init,
            is_cpu=False,
            name="relation_embedding")
        self.relation_feat = None

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

        self.loss_func = LossFunc(
            args,
            loss_type=args.loss_genre,
            neg_adv_sampling=args.neg_adversarial_sampling,
            adv_temp_value=args.adversarial_temperature,
            pairwise=args.pairwise)

    def head_forward(self, pos_triplets, neg_triplets, real_ent_ids):
        entity_emb = self.entity_embedding(real_ent_ids)
        if not self.args.cpu_emb:
            self.entity_embedding.curr_emb = entity_emb
        entity_feat = paddle.to_tensor(self.entity_feat[real_ent_ids.numpy()]
                                       .astype('float32'))
        emb = paddle.concat([entity_emb, entity_feat], axis=-1)

        pos_head = self.transform_net.embed_entity(
            F.embedding(pos_triplets[0], emb))
        pos_tail = self.transform_net.embed_entity(
            F.embedding(pos_triplets[2], emb))
        neg_head = self.transform_net.embed_entity(
            F.embedding(neg_triplets[0], emb))
        neg_tail = self.transform_net.embed_entity(
            F.embedding(neg_triplets[2], emb))
        pos_rel = self.transform_net.embed_relation(paddle.concat([self.relation_embedding(pos_triplets[1]), \
                                 paddle.to_tensor(self.relation_feat[pos_triplets[1].numpy()].astype('float32'))], axis=-1))
        neg_rel = self.transform_net.embed_relation(paddle.concat([self.relation_embedding(neg_triplets[1]), \
                                 paddle.to_tensor(self.relation_feat[neg_triplets[1].numpy()].astype('float32'))], axis=-1))
        batch_size = pos_head.shape[0]
        if batch_size < self.args.neg_sample_size:
            neg_sample_size = batch_size
        else:
            neg_sample_size = self.args.neg_sample_size
        pos_score = self.score_function.get_score(pos_head, pos_rel, pos_tail)
        neg_score = self.score_function.get_neg_score(
            pos_head, pos_rel, pos_tail, batch_size, neg_sample_size,
            neg_sample_size, True)
        loss = self.loss_func.get_total_loss(pos_score, neg_score)
        return loss

    def tail_forward(self, pos_triplets, neg_triplets, real_ent_ids):
        entity_emb = self.entity_embedding(real_ent_ids)
        if not self.args.cpu_emb:
            self.entity_embedding.curr_emb = entity_emb

        entity_feat = paddle.to_tensor(self.entity_feat[real_ent_ids.numpy()]
                                       .astype('float32'))
        emb = paddle.concat([entity_emb, entity_feat], axis=-1)

        pos_head = self.transform_net.embed_entity(
            F.embedding(pos_triplets[0], emb))
        pos_tail = self.transform_net.embed_entity(
            F.embedding(pos_triplets[2], emb))
        neg_head = self.transform_net.embed_entity(
            F.embedding(neg_triplets[0], emb))
        neg_tail = self.transform_net.embed_entity(
            F.embedding(neg_triplets[2], emb))

        pos_rel = self.transform_net.embed_relation(paddle.concat([self.relation_embedding(pos_triplets[1]), \
                                 paddle.to_tensor(self.relation_feat[pos_triplets[1].numpy()].astype('float32'))], axis=-1))
        neg_rel = self.transform_net.embed_relation(paddle.concat([self.relation_embedding(neg_triplets[1]), \
                                 paddle.to_tensor(self.relation_feat[neg_triplets[1].numpy()].astype('float32'))], axis=-1))
        batch_size = pos_head.shape[0]
        if batch_size < self.args.neg_sample_size:
            neg_sample_size = batch_size
        else:
            neg_sample_size = self.args.neg_sample_size
        pos_score = self.score_function.get_score(pos_head, pos_rel, pos_tail)
        neg_score = self.score_function.get_neg_score(
            pos_head, pos_rel, pos_tail, batch_size, neg_sample_size,
            neg_sample_size, False)
        loss = self.loss_func.get_total_loss(pos_score, neg_score)
        return loss

    def forward_test_wikikg(self, query, ans, candidate, mode='h,r->t'):
        scores = self.predict_wikikg_score(query, candidate, mode)
        argsort = paddle.argsort(scores, axis=1, descending=True)
        return argsort[:, :10], scores

    def predict_wikikg_score(self, query, candidate, mode):
        if mode == 'h,r->t':
            neg_sample_size = candidate.shape[1]
            neg_tail_entity_emb = self.entity_embedding(
                paddle.reshape(candidate, [-1]))
            neg_tail_entity_feat = paddle.to_tensor(self.entity_feat[
                paddle.reshape(candidate, [-1]).numpy()].astype('float32'))
            neg_tail = self.transform_net.embed_entity(
                paddle.concat([neg_tail_entity_emb, neg_tail_entity_feat], -1))

            head_emb = self.entity_embedding(query[:, 0])
            head_feat = paddle.to_tensor(self.entity_feat[query[:, 0].numpy()]
                                         .astype('float32'))
            head = self.transform_net.embed_entity(
                paddle.concat([head_emb, head_feat], -1))

            rel = self.transform_net.embed_relation(paddle.concat([self.relation_embedding(query[:, 1]), \
                                 paddle.to_tensor(self.relation_feat[query[:, 1].numpy()].astype('float32'))], axis=-1))
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
