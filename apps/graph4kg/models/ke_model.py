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

from models.numpy_embedding import NumPyEmbedding
from models.score_funcs import TransEScore, RotatEScore, DistMultScore, OTEScore
from utils import uniform, timer_wrapper


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


EMB_INIT_EPSILON = 2.


class KGEModel(nn.Layer):
    """
    Shallow model for knowledge representation learning.
    """

    @timer_wrapper('model construction')
    def __init__(self, model_name, trigraph, args=None):
        super(KGEModel, self).__init__()
        self._model_name = model_name
        mix_cpu_on_relation = False

        self._use_feat = args.use_feature
        self._ent_feat = None
        self._rel_feat = None
        if args.data_name == 'wikikg90m':
            if self._use_feat:
                self._ent_feat = trigraph.ent_feat.get('text')
                self._rel_feat = trigraph.rel_feat.get('text')

        self._mix_cpu_gpu = args.mix_cpu_gpu
        self._rel_emb_on_cpu = self._mix_cpu_gpu and mix_cpu_on_relation
        self._ent_emb_on_cpu = self._mix_cpu_gpu

        self._rel_times = args.rel_times
        self._ent_times = args.ent_times
        self._optim = 'adagrad' if args.mix_cpu_gpu else None
        self._lr = args.lr if args.mix_cpu_gpu else None
        if self._mix_cpu_gpu:
            print(('=' * 30) + '\n using cpu embeddings\n' + ('=' * 30))

        self._ent_dim = args.embed_dim * self._ent_times
        self._rel_dim = args.embed_dim * (
            self._rel_times + int(args.scale_type > 0))

        self.init_value = (args.gamma + EMB_INIT_EPSILON) / args.embed_dim

        self.ent_embedding = self._init_embedding(
            trigraph.num_ents, self._ent_dim, self.init_value,
            os.path.join(args.save_path, '__ent_embedding.npy')
            if self._ent_emb_on_cpu else None)

        self.rel_embedding = self._init_embedding(
            trigraph.num_rels, self._rel_dim, self.init_value,
            os.path.join(args.save_path, '__rel_embedding.npy')
            if self._rel_emb_on_cpu else None, args.scale_type)

        if self._use_feat:
            assert trigraph.ent_feat is not None, 'entity features not given!'
            assert trigraph.rel_feat is not None, 'relation features not given!'
            self._ent_feat_dim = self._ent_feat.shape[1]
            self._rel_feat_dim = self._rel_feat.shape[1]
            self.trans_ent = Transform(self._ent_feat_dim + self._ent_dim,
                                       self._ent_dim)
            self.trans_rel = Transform(self._rel_feat_dim + self._rel_dim,
                                       self._rel_dim)

        self._score_func = self._init_score_function(self._model_name.lower(),
                                                     args)

        # set parameters for chunk negative sampling
        self._num_chunks = 1
        if args.neg_sample_type == 'chunk':
            if args.neg_sample_size < args.batch_size:
                self._num_chunks = args.batch_size // args.neg_sample_size

    @property
    def shared_ent_path(self):
        if self._ent_emb_on_cpu:
            return self.ent_embedding.weight_path
        return None

    @property
    def shared_rel_path(self):
        if self._rel_emb_on_cpu:
            return self.rel_embedding.weight_path
        return None

    def train(self):
        if self._ent_emb_on_cpu:
            self.ent_embedding.train()
        if self._rel_emb_on_cpu:
            self.rel_embedding.train()

    def prepare_inputs(self,
                       h_index,
                       r_index,
                       t_index,
                       all_ent_index,
                       neg_ent_index=None,
                       ent_emb=None,
                       rel_emb=None,
                       mode='tail',
                       args=None):
        if ent_emb is not None:
            if self._use_feat:
                ent_feat = paddle.to_tensor(
                    self._ent_feat(all_ent_index.numpy()).astype('float32'))
                ent_emb = self.trans_ent(ent_feat, ent_emb)
        else:
            ent_emb = self._get_ent_embedding(all_ent_index)

        if rel_emb is not None:
            if self._use_feat:
                rel_feat = paddle.to_tensor(
                    self._rel_feat(r_index.numpy()).astype('float32'))
                pos_r = self.trans_rel(rel_feat, rel_emb)
        else:
            pos_r = self._get_rel_embedding(r_index)

        pos_h = F.embedding(h_index, ent_emb)
        pos_t = F.embedding(t_index, ent_emb)

        mask = None
        if neg_ent_index is not None:
            neg_ent_emb = F.embedding(neg_ent_index, ent_emb)
            neg_ent_emb = paddle.reshape(neg_ent_emb,
                                         (self._num_chunks, -1, self._ent_dim))
            if args.neg_deg_sample:
                if mode == 'head':
                    pos_emb = paddle.reshape(pos_h, (self._num_chunks, -1,
                                                     self._ent_dim))
                else:
                    pos_emb = paddle.reshape(pos_t, (self._num_chunks, -1,
                                                     self._ent_dim))
                chunk_size = pos_emb.shape[1]
                neg_sample_size = neg_ent_emb.shape[1]
                neg_ent_emb = paddle.concat([pos_emb, neg_ent_emb], axis=1)
                mask = paddle.ones(
                    [
                        self._num_chunks,
                        chunk_size * (neg_sample_size + chunk_size)
                    ],
                    dtype='float32')
                neg_sample_size = chunk_size + neg_sample_size
                mask[:, 0::(neg_sample_size + 1)] = 0.
                mask = paddle.reshape(mask, [self._num_chunks, chunk_size, -1])
        else:
            neg_ent_emb = None

        return pos_h, pos_r, pos_t, neg_ent_emb, mask

    def forward(self, h_emb, r_emb, t_emb):
        """function for training
        """
        self.train()

        score = self._score_func(h_emb, r_emb, t_emb)
        return score

    def get_neg_score(self,
                      ent_emb,
                      rel_emb,
                      neg_emb,
                      neg_head=False,
                      mask=None):
        """function to calculate scores of negative samples
        """
        ent_emb = paddle.reshape(ent_emb,
                                 (self._num_chunks, -1, self._ent_dim))
        rel_emb = paddle.reshape(rel_emb,
                                 (self._num_chunks, -1, self._rel_dim))

        if neg_head:
            h_emb = neg_emb
            t_emb = ent_emb
        else:
            h_emb = ent_emb
            t_emb = neg_emb

        score = self._score_func.get_neg_score(h_emb, rel_emb, t_emb, neg_head)

        if mask is not None:
            score = score * mask

        return score

    @paddle.no_grad()
    def predict(self, ent, rel, cand, mode='tail'):
        """function for prediction
        """
        if self._ent_emb_on_cpu:
            self.ent_embedding.eval()
        if self._rel_emb_on_cpu:
            self.rel_embedding.eval()

        num_cands = cand.shape[1]
        cand = paddle.reshape(cand, (-1, ))
        ent_emb = self._get_ent_embedding(ent)
        rel_emb = self._get_rel_embedding(rel)
        cand_emb = self._get_ent_embedding(cand)

        ent_emb = paddle.unsqueeze(ent_emb, axis=1)
        rel_emb = paddle.unsqueeze(rel_emb, axis=1)
        cand_emb = paddle.reshape(cand_emb, (-1, num_cands, self._ent_dim))
        cand_emb = cand_emb.tile((ent_emb.shape[0], 1, 1))

        if mode == 'tail':
            scores = self._score_func.get_neg_score(ent_emb, rel_emb, cand_emb,
                                                    False)
        else:
            scores = self._score_func.get_neg_score(cand_emb, rel_emb, ent_emb,
                                                    True)
        scores = paddle.squeeze(scores, axis=1)
        return scores

    def step(self, ent_trace=None, rel_trace=None):
        """Update NumPyEmbeddings
        """
        if self._ent_emb_on_cpu:
            if ent_trace is None:
                self.ent_embedding.step()
            else:
                self.ent_embedding.step_trace(ent_trace)
        else:
            if ent_trace is not None:
                raise ValueError(
                    "You are using gpu ent_emb, ent_trace must be None")

        if self._rel_emb_on_cpu:
            if rel_trace is None:
                self.rel_embedding.step()
            else:
                self.rel_embedding.step_trace(rel_trace)
        else:
            if rel_trace is not None:
                raise ValueError(
                    "You are using gpu rel_emb, rel_trace must be None")

    def start_async_update(self):
        """Initialize async update
        """
        if self._ent_emb_on_cpu:
            self.ent_embedding.start_async_update()
        if self._rel_emb_on_cpu:
            self.rel_embedding.start_async_update()

    def finish_async_update(self):
        """Finish async update
        """
        if self._ent_emb_on_cpu:
            self.ent_embedding.finish_async_update()
        if self._rel_emb_on_cpu:
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
                        emb_path=None,
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
            init_value = 2. / np.sqrt(emb_dim)
            a, b = -init_value, init_value

        if self._mix_cpu_gpu and emb_path is not None:
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

    def _init_score_function(self, model_name, args):
        if model_name == 'transe':
            score_func = TransEScore(args.gamma)
        elif model_name == 'rotate':
            score_func = RotatEScore(args.gamma, self.init_value)
        elif model_name == 'distmult':
            score_func = DistMultScore()
        elif model_name == 'ote':
            score_func = OTEScore(args.gamma, self._rel_times, args.scale_type)
        else:
            raise ValueError('score function for %s not implemented!' %
                             model_name)
        return score_func
