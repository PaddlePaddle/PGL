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

import os
import math

import paddle
import paddle.nn as nn
import numpy as np
import paddle.nn.functional as F

from models.embedding import NumPyEmbedding
from models.score_functions import TransEScore, RotatEScore, OTEScore
from utils.helper import uniform, timer_wrapper


class Transform(nn.Layer):
    """Transform model to combine embeddings and features
    """

    def __init__(self, in_dim, out_dim):
        super(Transform, self).__init__()
        init = np.sqrt(6. / (in_dim + out_dim))
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.Uniform(
            low=-init, high=init))
        self.linear = nn.Linear(in_dim, out_dim, weight_attr=weight_attr)

    def __call__(self, feats, embs):
        x = paddle.concat([feats, embs], axis=-1)
        return self.linear(x)


class KGEModel(nn.Layer):
    """
    Shallow model for knowledge representation learning.
    """

    @timer_wrapper('model construction')
    def __init__(self,
                 num_ents,
                 num_rels,
                 embed_dim,
                 score,
                 cpu_emb=False,
                 ent_times=1,
                 rel_times=1,
                 use_feat=False,
                 ent_feat=None,
                 rel_feat=None,
                 init_value=None,
                 scale_type=-1,
                 param_path='./',
                 optimizer='adagrad',
                 lr=1e-3,
                 args=None):
        super(KGEModel, self).__init__()
        self._use_feat = use_feat
        self._ent_feat = ent_feat
        self._rel_feat = rel_feat
        self._cpu_emb = cpu_emb
        self._rel_times = rel_times
        self._optim = optimizer
        self._lr = lr
        if self._cpu_emb:
            print(('=' * 30) + '\n using cpu embeddings\n' + ('=' * 30))

        self._ent_dim = embed_dim * ent_times
        self._rel_dim = embed_dim * (rel_times + int(scale_type > 0))
        self.ent_embedding = self._init_embedding(
            num_ents, self._ent_dim, init_value,
            os.path.join(param_path, '__ent_embedding.npy'))
        self.rel_embedding = self._init_embedding(
            num_rels, self._rel_dim, init_value,
            os.path.join(param_path, '__rel_embedding.npy'), scale_type)

        if self._use_feat:
            assert ent_feat is not None, 'entity features not given!'
            assert rel_feat is not None, 'relation features not given!'
            self._ent_feat_dim = self._ent_feat.shape[1]
            self._rel_feat_dim = self._rel_feat.shape[1]
            self.trans_ent = Transform(self._ent_feat_dim + self._ent_dim,
                                       self._ent_dim)
            self.trans_rel = Transform(self._rel_feat_dim + self._rel_dim,
                                       self._rel_dim)

        self._score_func = self._init_score_function(score.lower(), args)

    @property
    def shared_ent_path(self):
        if self._cpu_emb:
            return self.ent_embedding.weight_path
        return None

    @property
    def shared_rel_path(self):
        if self._cpu_emb:
            return self.rel_embedding.weight_path
        return None

    def forward(self,
                h,
                r,
                t,
                neg_ents,
                all_idxs,
                neg_mode='tail',
                r_emb=None,
                all_ent_emb=None):
        """function for training
        """
        if self._cpu_emb:
            self.ent_embedding.train()
            self.rel_embedding.train()
        if r_emb is not None and all_ent_emb is not None:
            ent_emb = all_ent_emb
            pos_r = r_emb
        else:
            ent_emb = self._get_ent_embedding(all_idxs)
            pos_r = self._get_rel_embedding(r)
        pos_h = paddle.unsqueeze(F.embedding(h, ent_emb), axis=1)
        pos_t = paddle.unsqueeze(F.embedding(t, ent_emb), axis=1)
        pos_r = paddle.unsqueeze(pos_r, axis=1)
        neg_ents_shape = neg_ents.shape
        neg_ents = F.embedding(paddle.reshape(neg_ents, (-1, )), ent_emb)
        neg_ents = paddle.reshape(neg_ents, [*neg_ents_shape, -1])

        pos_score = self._score_func(pos_h, pos_r, pos_t)
        if neg_mode == 'tail':
            neg_score = self._score_func(pos_h, pos_r, neg_ents)
        else:
            neg_score = self._score_func.inverse(neg_ents, pos_r, pos_t)

        return pos_score, neg_score

    @paddle.no_grad()
    def predict(self, ent, rel, cand, mode='tail'):
        """function for prediction
        """
        if self._cpu_emb:
            self.ent_embedding.eval()
            self.rel_embedding.eval()
        num_cands = cand.shape[1]
        cand = paddle.reshape(cand, (-1, ))
        ent_emb = self._get_ent_embedding(ent)
        rel_emb = self._get_rel_embedding(rel)
        cand_emb = self._get_ent_embedding(cand)

        ent_emb = paddle.unsqueeze(ent_emb, axis=1)
        rel_emb = paddle.unsqueeze(rel_emb, axis=1)
        cand_emb = paddle.reshape(cand_emb, (-1, num_cands, self._ent_dim))

        if mode == 'tail':
            scores = self._score_func(ent_emb, rel_emb, cand_emb)
        else:
            scores = self._score_func.inverse(cand_emb, rel_emb, ent_emb)
        scores = paddle.squeeze(scores, axis=1)
        return scores

    def step(self, ent_trace=None, rel_trace=None):
        """Update NumPyEmbeddings
        """
        if self._cpu_emb:
            if ent_trace is None:
                self.ent_embedding.step()
            else:
                self.ent_embedding.step_trace(ent_trace)
            if rel_trace is None:
                self.rel_embedding.step()
            else:
                self.rel_embedding.step_trace(rel_trace)

    def start_async_update(self):
        """Initialize async update
        """
        self.ent_embedding.start_async_update()
        self.rel_embedding.start_async_update()

    def finish_async_update(self):
        """Finish async update
        """
        self.ent_embedding.finish_async_update()
        self.rel_embedding.finish_async_update()

    def _get_ent_embedding(self, index):
        emb = self.ent_embedding(index)
        if self._use_feat:
            feat = paddle.to_tensor(
                self._ent_feat(index.numpy()).astype('float32'))
            emb = self.trans_ent(feat, emb)
        return emb

    def _get_rel_embedding(self, index):
        emb = self.rel_embedding(index)
        if self._use_feat:
            feat = paddle.to_tensor(
                self._rel_feat(index.numpy()).astype('float32'))
            emb = self.trans_rel(feat, emb)
        return emb

    def _init_embedding(self,
                        num_emb,
                        emb_dim,
                        init_value,
                        emb_path,
                        scale_type=-1,
                        name=None):
        if scale_type > 0:
            init_value = 1. / math.sqrt(emb_dim)
            a, b = -init_value, init_value
        elif isinstance(init_value, float):
            a, b = -init_value, init_value
        elif isinstance(init_value, tuple) or isinstance(init_value, list):
            assert len(init_value) == 2, 'invalid initialization range!'
            a, b = init_value
        else:
            init_value = 2. / emb_dim
            a, b = -init_value, init_value

        if self._cpu_emb:
            embs = NumPyEmbedding(num_emb, emb_dim, a, b, emb_path,
                                  self._optim, self._lr)
        else:
            weight_attr = paddle.ParamAttr(
                name=name, initializer=nn.initializer.Uniform(
                    low=a, high=b))
            embs = nn.Embedding(num_emb, emb_dim, weight_attr=weight_attr)

        if scale_type > 0:
            scale = int(scale_type == 1)
            embs.weight.reshape((-1, self._rel_times + 1))[:, -1] = scale

        return embs

    def _init_score_function(self, score, args):
        if score == 'transe':
            score_func = TransEScore(args.gamma)
        elif score == 'rotate':
            score_func = RotatEScore(args.gamma, 2. / self._ent_dim)
        elif score == 'ote':
            score_func = OTEScore(args.gamma, self._rel_times, args.scale_type)
        else:
            raise ValueError('score function for %s not implemented!' % score)
        return score_func
