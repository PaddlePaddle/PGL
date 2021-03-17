# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed und
# er the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
loader for the knowledge dataset.
"""
import os
import numpy as np
from collections import defaultdict
from pgl.utils.logger import log

#from pybloom import BloomFilter


class KGLoader:
    """
    load the FB15K
    """

    def __init__(self, data_dir, batch_size, neg_mode, neg_times):
        """init"""
        self.name = os.path.split(data_dir)[-1]
        self._feed_list = ["pos_triple", "neg_triple"]
        self._data_dir = data_dir
        self._batch_size = batch_size
        self._neg_mode = neg_mode
        self._neg_times = neg_times

        self._entity2id = {}
        self._relation2id = {}
        self.training_triple_pool = set()

        self._triple_train = None
        self._triple_test = None
        self._triple_valid = None

        self.entity_total = 0
        self.relation_total = 0
        self.train_num = 0
        self.test_num = 0
        self.valid_num = 0

        self.load_data()

    def test_data_batch(self, batch_size=None):
        """
        Test data reader.
        :param batch_size: Todo: batch_size > 1.
        :return: None
        """
        for i in range(self.test_num):
            data = np.array(self._triple_test[i])
            data = data.reshape((-1))
            yield [data]

    def training_data_no_filter(self, train_triple_positive):
        """faster, no filter for exists triples"""
        size = len(train_triple_positive) * self._neg_times
        train_triple_negative = train_triple_positive.repeat(
            self._neg_times, axis=0)
        replace_head_probability = 0.5 * np.ones(size)
        replace_entity_id = np.random.randint(self.entity_total, size=size)
        random_num = np.random.random(size=size)
        index_t = (random_num < replace_head_probability) * 1
        train_triple_negative[:, 0] = train_triple_negative[:, 0] + (
            replace_entity_id - train_triple_negative[:, 0]) * index_t
        train_triple_negative[:, 2] = replace_entity_id + (
            train_triple_negative[:, 2] - replace_entity_id) * index_t
        train_triple_positive = np.expand_dims(train_triple_positive, axis=2)
        train_triple_negative = np.expand_dims(train_triple_negative, axis=2)
        return train_triple_positive, train_triple_negative

    def training_data_map(self, train_triple_positive):
        """
        Map function for negative sampling.
        :param train_triple_positive: the triple positive.
        :return: the positive and negative triples.
        """
        size = len(train_triple_positive)
        train_triple_negative = []
        for i in range(size):
            corrupt_head_prob = np.random.binomial(1, 0.5)
            head_neg = train_triple_positive[i][0]
            relation = train_triple_positive[i][1]
            tail_neg = train_triple_positive[i][2]
            for j in range(0, self._neg_times):
                sample = train_triple_positive[i] + 0
                while True:
                    rand_id = np.random.randint(self.entity_total)
                    if corrupt_head_prob:
                        if (rand_id, relation, tail_neg
                            ) not in self.training_triple_pool:
                            sample[0] = rand_id
                            train_triple_negative.append(sample)
                            break
                    else:
                        if (head_neg, relation, rand_id
                            ) not in self.training_triple_pool:
                            sample[2] = rand_id
                            train_triple_negative.append(sample)
                            break
        train_triple_positive = np.expand_dims(train_triple_positive, axis=2)
        train_triple_negative = np.expand_dims(train_triple_negative, axis=2)
        if self._neg_mode:
            return train_triple_positive, train_triple_negative, np.array(
                [corrupt_head_prob], dtype="float32")
        return train_triple_positive, train_triple_negative

    def training_data_batch(self):
        """
        train_triple_positive
        :return:
        """
        n = len(self._triple_train)
        rand_idx = np.random.permutation(n)
        n_triple = len(rand_idx)
        start = 0
        while start < n_triple:
            end = min(start + self._batch_size, n_triple)
            train_triple_positive = self._triple_train[rand_idx[start:end]]
            start = end
            yield train_triple_positive

    def load_kg_triple(self, file):
        """
        Read in kg files.
        """
        triples = []
        with open(os.path.join(self._data_dir, file), "r") as f:
            for line in f.readlines():
                line_list = line.strip().split('\t')
                assert len(line_list) == 3
                head = self._entity2id[line_list[0]]
                tail = self._entity2id[line_list[1]]
                relation = self._relation2id[line_list[2]]
                triples.append((head, relation, tail))
        return np.array(triples)

    def load_data(self):
        """
        load kg dataset.
        """
        log.info("Start loading the {} dataset".format(self.name))
        with open(os.path.join(self._data_dir, 'entity2id.txt'), "r") as f:
            for line in f.readlines():
                line = line.strip().split('\t')
                self._entity2id[line[0]] = int(line[1])
        with open(os.path.join(self._data_dir, 'relation2id.txt'), "r") as f:
            for line in f.readlines():
                line = line.strip().split('\t')
                self._relation2id[line[0]] = int(line[1])
        self._triple_train = self.load_kg_triple('train.txt')
        self._triple_test = self.load_kg_triple('test.txt')
        self._triple_valid = self.load_kg_triple('valid.txt')

        self.relation_total = len(self._relation2id)
        self.entity_total = len(self._entity2id)
        self.train_num = len(self._triple_train)
        self.test_num = len(self._triple_test)
        self.valid_num = len(self._triple_valid)

        #bloom_capacity = len(self._triple_train) + len(self._triple_test) + len(self._triple_valid)
        #self.training_triple_pool = BloomFilter(capacity=bloom_capacity, error_rate=0.01)
        for i in range(len(self._triple_train)):
            self.training_triple_pool.add(
                (self._triple_train[i, 0], self._triple_train[i, 1],
                 self._triple_train[i, 2]))

        for i in range(len(self._triple_test)):
            self.training_triple_pool.add(
                (self._triple_test[i, 0], self._triple_test[i, 1],
                 self._triple_test[i, 2]))

        for i in range(len(self._triple_valid)):
            self.training_triple_pool.add(
                (self._triple_valid[i, 0], self._triple_valid[i, 1],
                 self._triple_valid[i, 2]))
        log.info('entity number: {}'.format(self.entity_total))
        log.info('relation number: {}'.format(self.relation_total))
        log.info('training triple number: {}'.format(self.train_num))
        log.info('testing triple number: {}'.format(self.test_num))
        log.info('valid triple number: {}'.format(self.valid_num))
