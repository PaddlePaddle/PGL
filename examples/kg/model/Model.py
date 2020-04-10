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
Base model of the knowledge graph embedding model.
"""
from paddle import fluid


class Model(object):
    """
    Base model.
    """

    def __init__(self, **kwargs):
        """
        Init model
        """
        # Needed parameters
        self.model_name = kwargs["model_name"]
        self.data_reader = kwargs["data_reader"]
        self._hidden_size = kwargs["hidden_size"]
        self._learning_rate = kwargs["learning_rate"]
        self._optimizer = kwargs["optimizer"]
        self.args = kwargs["args"]

        # Optional parameters
        if "margin" in kwargs:
            self._margin = kwargs["margin"]
        self._prefix = "%s_%s_dim=%d_" % (
            self.model_name, self.data_reader.name, self._hidden_size)
        self.ent_name = self._prefix + "entity_embeddings"
        self.rel_name = self._prefix + "relation_embeddings"

        self._entity_total = self.data_reader.entity_total
        self._relation_total = self.data_reader.relation_total
        self._ent_shape = [self._entity_total, self._hidden_size]
        self._rel_shape = [self._relation_total, self._hidden_size]

    def construct(self):
        """
        Construct the program
        :return: None
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
            self.train_feed_list = ["pos_triple", "neg_triple"]
            self.train_feed_vars = [self.train_pos_input, self.train_neg_input]
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

    def apply_optimizer(self, loss, opt="sgd"):
        """
        Construct the backward of the train program.
        :param loss: `type : variable` final loss of the model.
        :param opt: `type : string` the optimizer name
        :return:
        """
        optimizer_available = {
            "adam": fluid.optimizer.Adam,
            "sgd": fluid.optimizer.SGD,
            "momentum": fluid.optimizer.Momentum
        }
        if opt in optimizer_available:
            opt_func = optimizer_available[opt]
        else:
            opt_func = None
        if opt_func is None:
            raise ValueError("You should chose the optimizer in %s" %
                             optimizer_available.keys())
        else:
            optimizer = opt_func(learning_rate=self._learning_rate)
            return optimizer.minimize(loss)

    def construct_train_program(self):
        """
        This function should construct the train program with the `self.train_pos_input`
        and `self.train_neg_input`. These inputs are batch of triples.
        :return: List of variables you want to get. Please be sure the ':var loss' should
            be in the first place, eg. [loss, variable1, variable2, ...].
        """
        raise NotImplementedError(
            "You should define the construct_train_program"
            " function before use it!")

    def construct_test_program(self):
        """
        This function should construct test (or evaluate) program with the 'self.test_input'.
        Util now, we only support a triple the evaluate the ranks.
        :return: the distance of all entity with the test triple (for both head and tail entity).
        """
        raise NotImplementedError(
            "You should define the construct_test_program"
            " function before use it")
