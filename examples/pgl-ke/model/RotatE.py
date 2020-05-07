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
RotatE:
"RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space."
Sun, Zhiqing, et al.
https://arxiv.org/abs/1902.10197
"""
import paddle.fluid as fluid
from .Model import Model
from .utils import lookup_table


class RotatE(Model):
    """
    RotatE model.
    """

    def __init__(self,
                 data_reader,
                 hidden_size,
                 margin,
                 learning_rate,
                 args,
                 optimizer="adam"):
        super(RotatE, self).__init__(
            model_name="RotatE",
            data_reader=data_reader,
            hidden_size=hidden_size,
            margin=margin,
            learning_rate=learning_rate,
            args=args,
            optimizer=optimizer)

        self._neg_times = self.args.neg_times
        self._adv_temp_value = self.args.adv_temp_value

        self._relation_hidden_size = self._hidden_size
        self._entity_hidden_size = self._hidden_size * 2
        self._entity_embedding_margin = (
            self._margin + 2) / self._entity_hidden_size
        self._relation_embedding_margin = (
            self._margin + 2) / self._relation_hidden_size
        self._rel_shape = [self._relation_total, self._relation_hidden_size]
        self._ent_shape = [self._entity_total, self._entity_hidden_size]
        self._pi = 3.141592654

        self.construct_program()

    def construct_program(self):
        """
        construct the main program for train and test
        """
        self.startup_program = fluid.Program()
        self.train_program = fluid.Program()
        self.test_program = fluid.Program()

        with fluid.program_guard(self.train_program, self.startup_program):
            self.train_pos_input = fluid.layers.data(
                "pos_triple",
                dtype="int64",
                shape=[None, 3, 1],
                append_batch_size=False)
            self.train_neg_input = fluid.layers.data(
                "neg_triple",
                dtype="int64",
                shape=[None, 3, 1],
                append_batch_size=False)
            self.train_neg_mode = fluid.layers.data(
                "neg_mode",
                dtype='float32',
                shape=[1],
                append_batch_size=False)
            self.train_feed_vars = [
                self.train_pos_input, self.train_neg_input, self.train_neg_mode
            ]
            self.train_fetch_vars = self.construct_train_program()
            loss = self.train_fetch_vars[0]
            self.apply_optimizer(loss, opt=self._optimizer)

        with fluid.program_guard(self.test_program, self.startup_program):
            self.test_input = fluid.layers.data(
                "test_triple",
                dtype="int64",
                shape=[3],
                append_batch_size=False)
            self.test_feed_list = ["test_triple"]
            self.test_fetch_vars = self.construct_test_program()

    def creat_share_variables(self):
        """
        Share variables for train and test programs.
        """
        entity_embedding = fluid.layers.create_parameter(
            shape=self._ent_shape,
            dtype="float32",
            name=self.ent_name,
            default_initializer=fluid.initializer.Uniform(
                low=-1.0 * self._entity_embedding_margin,
                high=1.0 * self._entity_embedding_margin))
        relation_embedding = fluid.layers.create_parameter(
            shape=self._rel_shape,
            dtype="float32",
            name=self.rel_name,
            default_initializer=fluid.initializer.Uniform(
                low=-1.0 * self._relation_embedding_margin,
                high=1.0 * self._relation_embedding_margin))

        return entity_embedding, relation_embedding

    def score_with_l2_normalize(self, head, tail, rel, epsilon_var,
                                train_neg_mode):
        """
        Score function of RotatE  
        """
        one_var = fluid.layers.fill_constant(
            shape=[1], dtype='float32', value=1.0)
        re_head, im_head = fluid.layers.split(head, num_or_sections=2, dim=-1)
        re_tail, im_tail = fluid.layers.split(tail, num_or_sections=2, dim=-1)

        phase_relation = rel / (self._relation_embedding_margin / self._pi)
        re_relation = fluid.layers.cos(phase_relation)
        im_relation = fluid.layers.sin(phase_relation)

        re_score = re_relation * re_tail + im_relation * im_tail
        im_score = re_relation * im_tail - im_relation * re_tail
        re_score = re_score - re_head
        im_score = im_score - im_head
        #with fluid.layers.control_flow.Switch() as switch:
        #    with switch.case(train_neg_mode == one_var):
        #        re_score = re_relation * re_tail + im_relation * im_tail
        #        im_score = re_relation * im_tail - im_relation * re_tail
        #        re_score = re_score - re_head
        #        im_score = im_score - im_head
        #    with switch.default():
        #        re_score = re_head * re_relation - im_head * im_relation
        #        im_score = re_head * im_relation + im_head * re_relation
        #        re_score = re_score - re_tail
        #        im_score = im_score - im_tail

        re_score = re_score * re_score
        im_score = im_score * im_score

        score = re_score + im_score
        score = score + epsilon_var
        score = fluid.layers.sqrt(score)
        score = fluid.layers.reduce_sum(score, dim=-1)
        return self._margin - score

    def adverarial_weight(self, score):
        """
        adverarial the weight for softmax
        """
        adv_score = self._adv_temp_value * score
        adv_softmax = fluid.layers.softmax(adv_score)
        return adv_softmax

    def construct_train_program(self):
        """
        Construct train program
        """
        zero_var = fluid.layers.fill_constant(
            shape=[1], dtype='float32', value=0.0)
        epsilon_var = fluid.layers.fill_constant(
            shape=[1], dtype='float32', value=1e-12)
        entity_embedding, relation_embedding = self.creat_share_variables()
        pos_head = lookup_table(self.train_pos_input[:, 0], entity_embedding)
        pos_tail = lookup_table(self.train_pos_input[:, 2], entity_embedding)
        pos_rel = lookup_table(self.train_pos_input[:, 1], relation_embedding)
        neg_head = lookup_table(self.train_neg_input[:, 0], entity_embedding)
        neg_tail = lookup_table(self.train_neg_input[:, 2], entity_embedding)
        neg_rel = lookup_table(self.train_neg_input[:, 1], relation_embedding)

        pos_score = self.score_with_l2_normalize(pos_head, pos_tail, pos_rel,
                                                 epsilon_var, zero_var)
        neg_score = self.score_with_l2_normalize(
            neg_head, neg_tail, neg_rel, epsilon_var, self.train_neg_mode)

        neg_score = fluid.layers.reshape(
            neg_score, shape=[-1, self._neg_times], inplace=True)

        if self._adv_temp_value > 0.0:
            sigmoid_pos_score = fluid.layers.logsigmoid(1.0 * pos_score)
            sigmoid_neg_score = fluid.layers.logsigmoid(
                -1.0 * neg_score) * self.adverarial_weight(neg_score)
            sigmoid_neg_score = fluid.layers.reduce_sum(
                sigmoid_neg_score, dim=-1)
        else:
            sigmoid_pos_score = fluid.layers.logsigmoid(pos_score)
            sigmoid_neg_score = fluid.layers.logsigmoid(-1.0 * neg_score)

        loss_1 = fluid.layers.mean(sigmoid_pos_score)
        loss_2 = fluid.layers.mean(sigmoid_neg_score)
        loss = -1.0 * (loss_1 + loss_2) / 2
        return [loss]

    def score_with_l2_normalize_with_validate(self, entity_embedding, head,
                                              rel, tail, epsilon_var):
        """
        the score function for validation
        """
        re_entity_embedding, im_entity_embedding = fluid.layers.split(
            entity_embedding, num_or_sections=2, dim=-1)
        re_head, im_head = fluid.layers.split(head, num_or_sections=2, dim=-1)
        re_tail, im_tail = fluid.layers.split(tail, num_or_sections=2, dim=-1)
        phase_relation = rel / (self._relation_embedding_margin / self._pi)
        re_relation = fluid.layers.cos(phase_relation)
        im_relation = fluid.layers.sin(phase_relation)

        re_score = re_relation * re_tail + im_relation * im_tail
        im_score = re_relation * im_tail - im_relation * re_tail
        re_score = re_entity_embedding - re_score
        im_score = im_entity_embedding - im_score

        re_score = re_score * re_score
        im_score = im_score * im_score
        head_score = re_score + im_score
        head_score += epsilon_var
        head_score = fluid.layers.sqrt(head_score)
        head_score = fluid.layers.reduce_sum(head_score, dim=-1)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        re_score = re_entity_embedding - re_score
        im_score = im_entity_embedding - im_score

        re_score = re_score * re_score
        im_score = im_score * im_score
        tail_score = re_score + im_score
        tail_score += epsilon_var
        tail_score = fluid.layers.sqrt(tail_score)
        tail_score = fluid.layers.reduce_sum(tail_score, dim=-1)

        return head_score, tail_score

    def construct_test_program(self):
        """
        Construct test program
        """
        epsilon_var = fluid.layers.fill_constant(
            shape=[1], dtype='float32', value=1e-12)
        entity_embedding, relation_embedding = self.creat_share_variables()

        head_vec = lookup_table(self.test_input[0], entity_embedding)
        rel_vec = lookup_table(self.test_input[1], relation_embedding)
        tail_vec = lookup_table(self.test_input[2], entity_embedding)
        head_vec = fluid.layers.unsqueeze(head_vec, axes=[0])
        rel_vec = fluid.layers.unsqueeze(rel_vec, axes=[0])
        tail_vec = fluid.layers.unsqueeze(tail_vec, axes=[0])

        id_replace_head, id_replace_tail = self.score_with_l2_normalize_with_validate(
            entity_embedding, head_vec, rel_vec, tail_vec, epsilon_var)

        id_replace_head = fluid.layers.logsigmoid(id_replace_head)
        id_replace_tail = fluid.layers.logsigmoid(id_replace_tail)

        return [id_replace_head, id_replace_tail]
