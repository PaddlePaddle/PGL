# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""test_dataloader"""

import time
import unittest
import json
import os
import random

from pgl.utils.data.dataset import Dataset, StreamDataset
from pgl.utils.data.dataloader import Dataloader

DATA_SIZE = 20


class ListDataset(Dataset):
    def __init__(self):
        self.dataset = list(range(0, DATA_SIZE))

    def __getitem__(self, idx):
        return self._transform(self.dataset[idx])

    def __len__(self):
        return len(self.dataset)

    def _transform(self, example):
        time.sleep(0.05 + random.random() * 0.1)
        return example


class IterDataset(StreamDataset):
    def __init__(self):
        self.dataset = list(range(0, DATA_SIZE))

    def __iter__(self):
        for count, data in enumerate(self.dataset):
            if count % self._worker_info.num_workers != self._worker_info.fid:
                continue
            time.sleep(0.1)
            yield data


class Collate_fn(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, batch_examples):
        feed_dict = {}
        feed_dict['data'] = batch_examples
        feed_dict['labels'] = [i for i in range(len(batch_examples))]
        return feed_dict


class DataloaderTest(unittest.TestCase):
    def test_ListDataset(self):
        config = {
            'batch_size': 3,
            'drop_last': True,
            'shuffle': True,
            'num_workers': 2,
        }
        collate_fn = Collate_fn(config)
        ds = ListDataset()

        # test batch_size
        loader = Dataloader(
            ds,
            batch_size=config['batch_size'],
            drop_last=config['drop_last'],
            num_workers=config['num_workers'],
            collate_fn=collate_fn)

        epochs = 1
        for e in range(epochs):
            res = []
            for batch_data in loader:
                res.extend(batch_data['data'])
                self.assertEqual(len(batch_data['data']), config['batch_size'])

        # test shuffle
        loader = Dataloader(
            ds,
            batch_size=3,
            drop_last=False,
            shuffle=True,
            num_workers=1,
            collate_fn=collate_fn)

        for e in range(epochs):
            res = []
            for batch_data in loader:
                res.extend(batch_data['data'])
            self.assertEqual(set([i for i in range(DATA_SIZE)]), set(res))


    def test_IterDataset(self):
        config = {
            'batch_size': 3,
            'drop_last': True,
            'num_workers': 2,
        }
        collate_fn = Collate_fn(config)
        ds = IterDataset()
        loader = Dataloader(
            ds,
            batch_size=config['batch_size'],
            drop_last=config['drop_last'],
            num_workers=config['num_workers'],
            collate_fn=collate_fn)

        epochs = 1
        for e in range(epochs):
            res = []
            for batch_data in loader:
                res.extend(batch_data['data'])
                self.assertEqual(len(batch_data['data']), config['batch_size'])

        # test shuffle
        loader = Dataloader(
            ds,
            batch_size=3,
            drop_last=False,
            num_workers=1,
            collate_fn=collate_fn)

        for e in range(epochs):
            res = []
            for batch_data in loader:
                res.extend(batch_data['data'])
            self.assertEqual(set([i for i in range(DATA_SIZE)]), set(res))

    def test_ListDataset_Order(self):
        config = {
            'batch_size': 2,
            'drop_last': False,
            'shuffle': False,
            'num_workers': 4,
        }
        collate_fn = Collate_fn(config)
        ds = ListDataset()

        # test batch_size
        loader = Dataloader(
            ds,
            batch_size=config['batch_size'],
            drop_last=config['drop_last'],
            num_workers=config['num_workers'],
            collate_fn=collate_fn)

        epochs = 1
        for e in range(epochs):
            res = []
            for batch_data in loader:
                res.extend(batch_data['data'])
            self.assertEqual([i for i in range(DATA_SIZE)], res)


if __name__ == "__main__":
    unittest.main()
