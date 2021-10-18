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

import os
import paddle
import numpy as np
import unittest

from models.score_functions import TransEScore
from models.base_loss import LossFunction
from models.models import KGEModel


class AccuracyTest(unittest.TestCase):
    def test_transe_pos_score(self):
        score_func = TransEScore(gamma=19.9)
        np.random.seed(0)
        h = paddle.to_tensor(np.random.random((1000, 400)))
        t = paddle.to_tensor(np.random.random((1000, 400)))
        r = paddle.to_tensor(np.random.random((1000, 400)))
        pos_score = score_func(h, r, t).sum().numpy()
        trg_pos_score = 5781.9024776
        self.assertAlmostEqual(pos_score, trg_pos_score, None,
                               'pos_score not aligned!', 1e-6)

    def test_transe_neg_score(self):
        score_func = TransEScore(gamma=19.9)
        np.random.seed(0)
        h = paddle.to_tensor(np.random.random((1000, 400))).unsqueeze(0)
        t = paddle.to_tensor(np.random.random((1000, 400))).unsqueeze(0)
        r = paddle.to_tensor(np.random.random((1000, 400))).unsqueeze(0)
        neg_e = paddle.to_tensor(np.random.random((1000, 400))).unsqueeze(0)
        neg_score = score_func.multi_t(h, r, neg_e).sum().numpy()
        trg_neg_score = 5760053.6210946
        self.assertAlmostEqual(neg_score, trg_neg_score, None,
                               'neg_score not aligned!', 1e-6)

    def test_transe_loss(self):
        loss_func = LossFunction(
            name='Logsigmoid',
            pairwise=False,
            margin=1.0,
            neg_adv_spl=True,
            neg_adv_temp=1.0)
        np.random.seed(0)
        pos_score = paddle.to_tensor(np.random.random((1000, 1)))
        neg_score = paddle.to_tensor(np.random.random((1000, 1)))
        loss = loss_func(pos_score, neg_score).numpy()
        trg_loss = 0.7385364
        self.assertAlmostEqual(loss, trg_loss, None, 'loss_func not aligned!',
                               1e-6)

    def test_transe_grads(self):
        model = KGEModel(
            num_ents=3000,
            num_rels=1000,
            embed_dim=400,
            score='TransE',
            cpu_emb=False,
            init_value=1,
            use_feat=False,
            ent_feat=None,
            rel_feat=None,
            ent_times=1,
            rel_times=1,
            scale_type=0,
            param_path=None,
            optimizer='adagrad',
            lr=0.25)
        loss_func = LossFunction(
            name='Logsigmoid',
            pairwise=False,
            margin=1.0,
            neg_adv_spl=True,
            neg_adv_temp=1.0)
        np.random.seed(0)
        all_ents_emb = paddle.to_tensor(np.random.random((3000, 400)))
        r_emb = paddle.to_tensor(np.random.random((1000, 400)))
        all_ents_emb.stop_gradient = False
        r_emb.stop_gradient = False
        h = paddle.to_tensor([x for x in range(1000)])
        t = paddle.to_tensor([x for x in range(1000, 2000)])
        neg_ents = paddle.to_tensor([x for x in range(2000, 3000)])
        all_ents = paddle.to_tensor([x for x in range(3000)])
        pos_score, neg_score = model(h, r, t, neg_ents, all_ents, 'tail',
                                     r_emb, all_ents_emb)
        loss = loss_func(pos_score, neg_score).numpy()
        loss.backward()
        ent_grad = all_ents_emb.grad.sum().numpy()
        rel_grad = r_emb.grad.sum().numpy()
        self.assertAlmostEqual(ent_grad, 0, None, 'ent_grad not aligned!',
                               1e-6)
        self.assertAlmostEqual(rel_grad, 0, None, 'rel_grad not aligned!',
                               1e-6)


if __name__ == '__main__':
    unittest.main()
