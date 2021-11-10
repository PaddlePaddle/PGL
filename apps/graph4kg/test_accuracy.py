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
import argparse
import paddle
import numpy as np
import unittest

from models.score_funcs import TransEScore, RotatEScore, QuatEScore, OTEScore
from models.loss_func import LossFunction
from models.ke_model import KGEModel
from dataset.trigraph import TriGraph


class AccuracyTest(unittest.TestCase):
    """Test Model Accuracy
    """

    def test_quate_initial(self):
        """test get_init_weight of quate
        """
        score_func = QuatEScore(1000)
        weight = score_func.get_init_weight(1000, 400)
        self.assertAlmostEqual(weight.mean(), 3.6877538962929613e-06, None,
                               'init mean not aligned!', 5e-7)
        self.assertAlmostEqual(weight.min(), -0.022353720820485274, None,
                               'init min not aligned!', 5e-5)
        self.assertAlmostEqual(weight.max(), 0.022350550943842893, None,
                               'init max not aligned!', 5e-5)
        self.assertAlmostEqual(weight.sum(), 5.900406234068738, None,
                               'init sum not aligned!', 1)

    def test_ote_pos_score(self):
        """test forward of ote
        """
        score_func = OTEScore(8, 4, 2)
        np.random.seed(0)
        h = paddle.to_tensor(np.random.random((10, 100)))
        t = paddle.to_tensor(np.random.random((10, 100)))
        r = paddle.to_tensor(np.random.random((10, 500)))
        score = score_func(h, r, t)
        print('ote score mean:', score.mean())
        print('ote score sum:', score.sum())

    def test_quate_forward(self):
        """test forward of quate
        """
        score_func = QuatEScore(1000)
        np.random.seed(0)
        embeds = []
        for i in range(12):
            embeds.append(np.random.random((1000, 100)))
        head = paddle.to_tensor(np.concatenate(embeds[0:4], axis=-1))
        tail = paddle.to_tensor(np.concatenate(embeds[4:8], axis=-1))
        rel = paddle.to_tensor(np.concatenate(embeds[8:12], axis=-1))
        score = score_func(head, rel, tail).cpu().detach().numpy()
        self.assertAlmostEqual(score.sum(), -44251.7816, None,
                               'forward sum not aligned!', 100)
        self.assertAlmostEqual(score.mean(), -44.2518, None,
                               'forward mean not aligned!', 0.1)

    def test_transe_pos_score(self):
        """test forward of transe
        """
        score_func = TransEScore(gamma=19.9)
        np.random.seed(0)
        h = paddle.to_tensor(np.random.random((1000, 400)))
        t = paddle.to_tensor(np.random.random((1000, 400)))
        r = paddle.to_tensor(np.random.random((1000, 400)))
        pos_score = score_func(h, r, t).sum().numpy()
        trg_pos_score = 5781.9024776
        self.assertAlmostEqual(pos_score, trg_pos_score, None,
                               'pos_score not aligned!', 5e-3)

    def test_transe_neg_score(self):
        """test get_neg_score of transe
        """
        score_func = TransEScore(gamma=19.9)
        np.random.seed(0)
        h = paddle.to_tensor(np.random.random((1000, 400))).unsqueeze(0)
        t = paddle.to_tensor(np.random.random((1000, 400))).unsqueeze(0)
        r = paddle.to_tensor(np.random.random((1000, 400))).unsqueeze(0)
        neg_e = paddle.to_tensor(np.random.random((1000, 400))).unsqueeze(0)
        neg_score = score_func.get_neg_score(h, r, neg_e, False).sum().numpy()
        trg_neg_score = 5760053.6210946
        self.assertAlmostEqual(neg_score, trg_neg_score, None,
                               'neg_score not aligned!', 0.5)

    def test_rotate_pos_score(self):
        """test forward of rotate
        """
        score_func = RotatEScore(gamma=12.0, embed_dim=200)
        np.random.seed(0)
        h = paddle.to_tensor(np.random.random((10, 400)))
        t = paddle.to_tensor(np.random.random((10, 400)))
        r = paddle.to_tensor(np.random.random((10, 200)))
        score = score_func(h, r, t).sum().numpy()
        trg_score = -1957.4883
        self.assertAlmostEqual(score, trg_score, None,
                               'pos_score not aligned!', 1e-4)

    def test_rotate_neg_score(self):
        """test get_neg_score of rotate
        """
        score_func = RotatEScore(gamma=12.0, embed_dim=200)
        np.random.seed(0)
        h = paddle.to_tensor(np.random.random((10, 400)))
        t = paddle.to_tensor(np.random.random((10, 400)))
        r = paddle.to_tensor(np.random.random((10, 200)))
        h = paddle.reshape(h, [2, 5, 400])
        t = paddle.reshape(t, [2, 5, 400])
        r = paddle.reshape(r, [2, 5, 200])
        score = score_func.get_neg_score(h, r, t, False).sum().numpy()
        trg_score = -9693.6280
        self.assertAlmostEqual(score, trg_score, None,
                               'pos_score not aligned!', 1e-4)

    def test_logsigmoid_loss(self):
        """test logsigmoid
        """
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
        self.assertAlmostEqual(loss, trg_loss, None,
                               'logsigmoid loss not aligned!', 1e-6)

    def test_softplus_loss(self):
        """test softplus
        """
        loss_func = LossFunction(
            name='Softplus',
            margin=0.0,
            pairwise=False,
            neg_adv_spl=False,
            neg_adv_temp=0.0)
        np.random.seed(0)
        pos_score = paddle.to_tensor(np.random.random((1000)))
        neg_score = paddle.to_tensor(np.random.random((1000, 100)))
        loss = loss_func(pos_score, neg_score).numpy()
        print('softplus loss', loss)


if __name__ == '__main__':
    unittest.main()
