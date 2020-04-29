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
TransR:
"Learning entity and relation embeddings for knowledge graph completion."
Lin, Yankai, et al.
https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9571/9523
"""
import numpy as np
import paddle.fluid as fluid
from .Model import Model
from .utils import lookup_table


class TransR(Model):
    """
    TransR model.
    """

    def __init__(self,
                 data_reader,
                 hidden_size,
                 margin,
                 learning_rate,
                 args,
                 optimizer="adam"):
        """init"""
        self._neg_times = args.neg_times
        super(TransR, self).__init__(
            model_name="TransR",
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
            shape=self._ent_shape,
            dtype="float32",
            name=self.ent_name,
            default_initializer=fluid.initializer.Xavier())
        relation_embedding = fluid.layers.create_parameter(
            shape=self._rel_shape,
            dtype="float32",
            name=self.rel_name,
            default_initializer=fluid.initializer.Xavier())
        init_values = np.tile(
            np.identity(
                self._hidden_size, dtype="float32").reshape(-1),
            (self._relation_total, 1))
        transfer_matrix = fluid.layers.create_parameter(
            shape=[
                self._relation_total, self._hidden_size * self._hidden_size
            ],
            dtype="float32",
            name=self._prefix + "transfer_matrix",
            default_initializer=fluid.initializer.NumpyArrayInitializer(
                init_values))

        return entity_embedding, relation_embedding, transfer_matrix

    def score_with_l2_normalize(self, head, rel, tail):
        """
        Score function of TransR
        """
        head = fluid.layers.l2_normalize(head, axis=-1)
        rel = fluid.layers.l2_normalize(rel, axis=-1)
        tail = fluid.layers.l2_normalize(tail, axis=-1)
        score = head + rel - tail
        return score

    @staticmethod
    def matmul_with_expend_dims(x, y):
        """matmul_with_expend_dims"""
        x = fluid.layers.unsqueeze(x, axes=[1])
        res = fluid.layers.matmul(x, y)
        return fluid.layers.squeeze(res, axes=[1])

    def construct_train_program(self):
        """
        Construct train program
        """
        entity_embedding, relation_embedding, transfer_matrix = self.creat_share_variables(
        )
        pos_head = lookup_table(self.train_pos_input[:, 0], entity_embedding)
        pos_tail = lookup_table(self.train_pos_input[:, 2], entity_embedding)
        pos_rel = lookup_table(self.train_pos_input[:, 1], relation_embedding)
        neg_head = lookup_table(self.train_neg_input[:, 0], entity_embedding)
        neg_tail = lookup_table(self.train_neg_input[:, 2], entity_embedding)
        neg_rel = lookup_table(self.train_neg_input[:, 1], relation_embedding)

        rel_matrix = fluid.layers.reshape(
            lookup_table(self.train_pos_input[:, 1], transfer_matrix),
            [-1, self._hidden_size, self._hidden_size])
        pos_head_trans = self.matmul_with_expend_dims(pos_head, rel_matrix)
        pos_tail_trans = self.matmul_with_expend_dims(pos_tail, rel_matrix)

        trans_neg = True
        if trans_neg:
            rel_matrix_neg = fluid.layers.reshape(
                lookup_table(self.train_neg_input[:, 1], transfer_matrix),
                [-1, self._hidden_size, self._hidden_size])
            neg_head_trans = self.matmul_with_expend_dims(neg_head,
                                                          rel_matrix_neg)
            neg_tail_trans = self.matmul_with_expend_dims(neg_tail,
                                                          rel_matrix_neg)
        else:
            neg_head_trans = self.matmul_with_expend_dims(neg_head, rel_matrix)
            neg_tail_trans = self.matmul_with_expend_dims(neg_tail, rel_matrix)

        pos_score = self.score_with_l2_normalize(pos_head_trans, pos_rel,
                                                 pos_tail_trans)
        neg_score = self.score_with_l2_normalize(neg_head_trans, neg_rel,
                                                 neg_tail_trans)

        pos = fluid.layers.reduce_sum(
            fluid.layers.abs(pos_score), -1, keep_dim=False)
        neg = fluid.layers.reduce_sum(
            fluid.layers.abs(neg_score), -1, keep_dim=False)
        neg = fluid.layers.reshape(
            neg, shape=[-1, self._neg_times], inplace=True)

        loss = fluid.layers.reduce_mean(
            fluid.layers.relu(pos - neg + self._margin))
        return [loss]

    def construct_test_program(self):
        """
        Construct test program
        """
        entity_embedding, relation_embedding, transfer_matrix = self.creat_share_variables(
        )
        rel_matrix = fluid.layers.reshape(
            lookup_table(self.test_input[1], transfer_matrix),
            [self._hidden_size, self._hidden_size])
        entity_embedding_trans = fluid.layers.matmul(entity_embedding,
                                                     rel_matrix, False, False)
        rel_vec = lookup_table(self.test_input[1], relation_embedding)
        entity_embedding_trans = fluid.layers.l2_normalize(
            entity_embedding_trans, axis=-1)
        rel_vec = fluid.layers.l2_normalize(rel_vec, axis=-1)
        head_vec = lookup_table(self.test_input[0], entity_embedding_trans)
        tail_vec = lookup_table(self.test_input[2], entity_embedding_trans)

        # The paddle fluid.layers.topk GPU OP is very inefficient
        # we do sort operation in the evaluation step using multiprocessing
        id_replace_head = fluid.layers.reduce_sum(
            fluid.layers.abs(entity_embedding_trans + rel_vec - tail_vec),
            dim=1)
        id_replace_tail = fluid.layers.reduce_sum(
            fluid.layers.abs(entity_embedding_trans - rel_vec - head_vec),
            dim=1)

        return [id_replace_head, id_replace_tail]
