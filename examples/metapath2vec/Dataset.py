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
This file loads and preprocesses the dataset for metapath2vec model.
"""

import sys
import os
import glob
import numpy as np
import tqdm
import time
import logging
import random
from pgl import heter_graph
import pickle as pkl


class Dataset(object):
    """Implementation of Dataset class

    This is a simple implementation of loading and processing dataset for metapath2vec model.

    Args:
        config: dict, some configure parameters.
    """

    NEGATIVE_TABLE_SIZE = 1e8

    def __init__(self, config):
        self.config = config
        self.walk_files = os.path.join(config['input_path'],
                                       config['walk_path'])
        self.word2id_file = os.path.join(config['input_path'],
                                         config['word2id_file'])

        self.word2freq = {}
        self.word2id = {}
        self.id2word = {}
        self.sentences_count = 0
        self.token_count = 0
        self.negatives = []
        self.discards = []

        logging.info('reading sentences')
        self.read_words()
        logging.info('initializing discards')
        self.initDiscards()
        logging.info('initializing negatives')
        self.initNegatives()

    def read_words(self):
        """Read words(nodes) from walk files which are produced by sampler.
        """
        word_freq = dict()
        for walk_file in glob.glob(self.walk_files):
            with open(walk_file, 'r') as reader:
                for walk in reader:
                    walk = walk.strip().split()
                    if len(walk) > 1:
                        self.sentences_count += 1
                        for word in walk:
                            if int(word) >= self.config[
                                    'paper_start_index']:  # remove paper
                                continue
                            else:
                                self.token_count += 1
                                word_freq[word] = word_freq.get(word, 0) + 1

        wid = 0
        logging.info('Read %d sentences.' % self.sentences_count)
        logging.info('Read %d words.' % self.token_count)
        logging.info('%d words have been sampled.' % len(word_freq))
        for w, c in word_freq.items():
            if c < self.config['min_count']:
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word2freq[wid] = c
            wid += 1

        self.word_count = len(self.word2id)
        logging.info(
            '%d words displayed less than %d(min_count) have been discarded.' %
            (len(word_freq) - len(self.word2id), self.config['min_count']))

        pkl.dump(self.word2id, open(self.word2id_file, 'wb'))

    def initDiscards(self):
        """Get a frequency table for sub-sampling.
        """
        t = 0.0001
        f = np.array(list(self.word2freq.values())) / self.token_count
        self.discards = np.sqrt(t / f) + (t / f)

    def initNegatives(self):
        """Get a table for negative sampling
        """
        pow_freq = np.array(list(self.word2freq.values()))**0.75
        words_pow = sum(pow_freq)
        ratio = pow_freq / words_pow
        count = np.round(ratio * Dataset.NEGATIVE_TABLE_SIZE)
        for wid, c in enumerate(count):
            self.negatives += [wid] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)
        self.sampling_prob = ratio

    def getNegatives(self, size):
        """Get negative samples from negative samling table.
        """
        return np.random.choice(self.negatives, size)

    def walk_from_files(self, walkpath_files):
        """Generate walks from files.
        """
        bucket = []
        for filename in walkpath_files:
            with open(filename) as reader:
                for line in reader:
                    words = line.strip().split()
                    words = [
                        w for w in words
                        if int(w) < self.config['paper_start_index']
                    ]
                    if len(words) > 1:
                        word_ids = [
                            self.word2id[w] for w in words if w in self.word2id
                        ]
                        bucket.append(word_ids)
                        if len(bucket) == self.config['batch_size']:
                            yield bucket
                            bucket = []
        if len(bucket):
            yield bucket

    def pairs_generator(self, walkpath_files):
        """Generate train pairs(src, pos, negs) for training model.
        """

        def wrapper():
            """wrapper for multiprocess calling.
            """
            for walks in self.walk_from_files(walkpath_files):
                res = self.gen_pairs(walks)
                yield res

        return wrapper

    def gen_pairs(self, walks):
        """Generate train pairs data for training model.
        """
        src = []
        pos = []
        negs = []
        skip_window = self.config['win_size'] // 2
        for walk in walks:
            for i in range(len(walk)):
                for j in range(1, skip_window + 1):
                    if i - j >= 0:
                        src.append(walk[i])
                        pos.append(walk[i - j])
                        negs.append(
                            self.getNegatives(size=self.config['neg_num']))
                    if i + j < len(walk):
                        src.append(walk[i])
                        pos.append(walk[i + j])
                        negs.append(
                            self.getNegatives(size=self.config['neg_num']))

        src = np.array(src, dtype=np.int64).reshape(-1, 1, 1)
        pos = np.array(pos, dtype=np.int64).reshape(-1, 1, 1)
        negs = np.expand_dims(np.array(negs, dtype=np.int64), -1)
        return {"src": src, "pos": pos, "negs": negs}


if __name__ == "__main__":
    config = {
        'input_path': './data/out_aminer_CPAPC/',
        'walk_path': 'aminer_walks_CPAPC_500num_100len/*',
        'author_label_file': 'author_label.txt',
        'venue_label_file': 'venue_label.txt',
        'remapping_author_label_file': 'multi_class_author_label.txt',
        'remapping_venue_label_file': 'multi_class_venue_label.txt',
        'word2id_file': 'word2id.pkl',
        'win_size': 7,
        'neg_num': 5,
        'min_count': 2,
        'batch_size': 1,
    }

    log_format = '%(asctime)s-%(levelname)s-%(name)s: %(message)s'
    logging.basicConfig(level=getattr(logging, 'INFO'), format=log_format)

    dataset = Dataset(config)
