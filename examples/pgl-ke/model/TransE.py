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
"""
TransE:
"Translating embeddings for modeling multi-relational data."
Bordes, Antoine, et al.
https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf
"""
import paddle.fluid as fluid
from .Model import Model
from .utils import lookup_table


class TransE(Model):
    """
    The TransE Model.
    """

    def __init__(self,
                 data_reader,
                 hidden_size,
                 margin,
                 learning_rate,
                 args,
                 optimizer="adam"):
        self._neg_times = args.neg_times
        super(TransE, self).__init__(
            model_name="TransE",
            data_reader=data_reader,
            hidden_size=hidden_size,
            margin=margin,
            learning_rate=learning_rate,
            args=args,
            optimizer=optimizer)
        self.construct()

    def creat_share_variables(self):
        """
        Share variables for train and test programs.
        """
        entity_embedding = fluid.layers.create_parameter(
            shape=self._ent_shape, dtype="float32", name=self.ent_name)
        relation_embedding = fluid.layers.create_parameter(
            shape=self._rel_shape, dtype="float32", name=self.rel_name)
        return entity_embedding, relation_embedding

    @staticmethod
    def score_with_l2_normalize(head, rel, tail):
        """
        Score function of TransE
        """
        head = fluid.layers.l2_normalize(head, axis=-1)
        rel = fluid.layers.l2_normalize(rel, axis=-1)
        tail = fluid.layers.l2_normalize(tail, axis=-1)
        score = head + rel - tail
        return score

    def construct_train_program(self):
        """
        Construct train program.
        """
        entity_embedding, relation_embedding = self.creat_share_variables()
        pos_head = lookup_table(self.train_pos_input[:, 0], entity_embedding)
        pos_tail = lookup_table(self.train_pos_input[:, 2], entity_embedding)
        pos_rel = lookup_table(self.train_pos_input[:, 1], relation_embedding)
        neg_head = lookup_table(self.train_neg_input[:, 0], entity_embedding)
        neg_tail = lookup_table(self.train_neg_input[:, 2], entity_embedding)
        neg_rel = lookup_table(self.train_neg_input[:, 1], relation_embedding)

        pos_score = self.score_with_l2_normalize(pos_head, pos_rel, pos_tail)
        neg_score = self.score_with_l2_normalize(neg_head, neg_rel, neg_tail)

        pos = fluid.layers.reduce_sum(
            fluid.layers.abs(pos_score), 1, keep_dim=False)
        neg = fluid.layers.reduce_sum(
            fluid.layers.abs(neg_score), 1, keep_dim=False)
        neg = fluid.layers.reshape(
            neg, shape=[-1, self._neg_times], inplace=True)

        loss = fluid.layers.reduce_mean(
            fluid.layers.relu(pos - neg + self._margin))
        return [loss]

    def construct_test_program(self):
        """
        Construct test program
        """
        entity_embedding, relation_embedding = self.creat_share_variables()
        entity_embedding = fluid.layers.l2_normalize(entity_embedding, axis=-1)
        relation_embedding = fluid.layers.l2_normalize(
            relation_embedding, axis=-1)
        head_vec = lookup_table(self.test_input[0], entity_embedding)
        rel_vec = lookup_table(self.test_input[1], relation_embedding)
        tail_vec = lookup_table(self.test_input[2], entity_embedding)
        # The paddle fluid.layers.topk GPU OP is very inefficient
        # we do sort operation in the evaluation step using multiprocessing.
        id_replace_head = fluid.layers.reduce_sum(
            fluid.layers.abs(entity_embedding + rel_vec - tail_vec), dim=1)
        id_replace_tail = fluid.layers.reduce_sum(
            fluid.layers.abs(entity_embedding - rel_vec - head_vec), dim=1)

        return [id_replace_head, id_replace_tail]
