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

import paddle
from paddle.nn import BCELoss, MarginRankingLoss
from paddle.nn.functional import log_sigmoid, softmax, softplus


class LogSigmoidLoss(object):
    """LogSigmoidLoss.
    """

    def __init__(self):
        super(LogSigmoidLoss, self).__init__()
        self.neg_label = -1

    def __call__(self, score, label, weights):
        if weights is not None:
            if len(score.shape) > len(weights.shape):
                weights = weights.unsqueeze(1)
            return -(weights * log_sigmoid(label *
                                           score)).sum() / weights.sum()
        else:
            return -log_sigmoid(label * score)


class SoftplusLoss(object):
    """SoftplusLoss.
    """

    def __init__(self):
        super(SoftplusLoss, self).__init__()
        self.neg_label = -1

    def __call__(self, score, label):
        return -softplus(label * score)


class LossFunction(object):
    """Loss functions.

    Args:
        name (str):
            Loss function name.
        margin (float, optional):
            Margin parameter for MarginRankingLoss.
        pairwise (bool, optional):
            Whether compute loss of positive and negative samples pairwise.
        neg_adv_spl (bool, optional):
            Whether use negative adversarial sampling.
        neg_adv_temp (float, optional):
            Temperature of negative adversarial sampling.
    """

    def __init__(self,
                 name='LogSigmoid',
                 margin=4.0,
                 pairwise=False,
                 neg_adv_spl=False,
                 neg_adv_temp=0.0):
        super(LossFunction, self).__init__()
        self.name = name
        self.margin = margin
        self.pairwise = pairwise
        self.loss_func = self.get_loss_func()

        self.neg_adv_spl = neg_adv_spl
        self.neg_adv_temp = neg_adv_temp

    def __call__(self, pos_score, neg_score, weights=None):

        if self.pairwise:
            pos_score = paddle.unsqueeze(pos_score, -1)
            loss = paddle.mean(self.loss_func(pos_score - neg_score, 1))
        else:
            if self.name == 'Logsigmoid':
                pos_loss = self.loss_func(pos_score, 1, weights)
                neg_loss = self.loss_func(neg_score, self.neg_label, weights)
            else:
                pos_loss = self.loss_func(pos_score, 1)
                neg_loss = self.loss_func(neg_score, self.neg_label)
            if self.neg_adv_spl:
                neg_loss = neg_loss * self.adverarial_weight(neg_score)
                neg_loss = paddle.sum(neg_loss, axis=-1)
            else:
                neg_loss = paddle.mean(neg_loss, axis=-1)

            pos_loss = paddle.mean(pos_loss)
            neg_loss = paddle.mean(neg_loss)
            loss = (pos_loss + neg_loss) / 2
        return loss

    def adverarial_weight(self, score):
        """Adverarial weights for softmax.
        """
        adv_score = self.neg_adv_temp * score
        adv_softmax = softmax(adv_score, axis=-1)
        adv_softmax.stop_gradient = True
        return adv_softmax

    def get_loss_func(self):
        """Return loss function objective.
        """
        if self.name == 'Hinge':
            self.neg_label = -1
            return MarginRankingLoss(margin=self.margin)
        elif self.name == 'Logsigmoid':
            self.neg_label = -1
            return LogSigmoidLoss()
        elif self.name == 'BCE':
            self.neg_label = 0
            return BCELoss()
        elif self.name == 'Softplus':
            self.neg_label = -1
            return SoftplusLoss()
        else:
            raise ValueError('loss %s not implemented!' % self.name)
