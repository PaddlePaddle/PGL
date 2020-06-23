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
This file implement the graph dataloader.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# SSL

import torch
import sys
import six
from io import open
import collections
from collections import namedtuple
import numpy as np
import tqdm
import time

import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as fl
import pgl
from pgl.utils import mp_reader
from pgl.utils.logger import log

from ogb.graphproppred import GraphPropPredDataset


def batch_iter(data, batch_size, fid, num_workers):
    """node_batch_iter
    """
    size = len(data)
    perm = np.arange(size)
    np.random.shuffle(perm)
    start = 0
    cc = 0
    while start < size:
        index = perm[start:start + batch_size]
        start += batch_size
        cc += 1
        if cc % num_workers != fid:
            continue
        yield data[index]


def scan_batch_iter(data, batch_size, fid, num_workers):
    """scan_batch_iter
    """
    batch = []
    cc = 0
    for line_example in data.scan():
        cc += 1
        if cc % num_workers != fid:
            continue
        batch.append(line_example)
        if len(batch) == batch_size:
            yield batch
            batch = []

    if len(batch) > 0:
        yield batch


class GraphDataloader(object):
    """Graph Dataloader
    """

    def __init__(self,
                 dataset,
                 graph_wrapper,
                 batch_size,
                 seed=0,
                 num_workers=1,
                 buf_size=1000,
                 shuffle=True):

        self.shuffle = shuffle
        self.seed = seed
        self.num_workers = num_workers
        self.buf_size = buf_size
        self.batch_size = batch_size
        self.dataset = dataset
        self.graph_wrapper = graph_wrapper

    def batch_fn(self, batch_examples):
        """ batch_fn batch producer"""
        graphs = [b[0] for b in batch_examples]
        labels = [b[1] for b in batch_examples]
        join_graph = pgl.graph.MultiGraph(graphs)
        labels = np.array(labels)

        feed_dict = self.graph_wrapper.to_feed(join_graph)
        batch_valid = (labels == labels).astype("float32")
        labels = np.nan_to_num(labels).astype("float32")
        feed_dict['labels'] = labels
        feed_dict['unmask'] = batch_valid
        return feed_dict

    def batch_iter(self, fid):
        """batch_iter"""
        if self.shuffle:
            for batch in batch_iter(self, self.batch_size, fid,
                                    self.num_workers):
                yield batch
        else:
            for batch in scan_batch_iter(self, self.batch_size, fid,
                                         self.num_workers):
                yield batch

    def __len__(self):
        """__len__"""
        return len(self.dataset)

    def __getitem__(self, idx):
        """__getitem__"""
        if isinstance(idx, collections.Iterable):
            return [self[bidx] for bidx in idx]
        else:
            return self.dataset[idx]

    def __iter__(self):
        """__iter__"""

        def worker(filter_id):
            def func_run():
                for batch_examples in self.batch_iter(filter_id):
                    batch_dict = self.batch_fn(batch_examples)
                    yield batch_dict

            return func_run

        if self.num_workers == 1:
            r = paddle.reader.buffered(worker(0), self.buf_size)
        else:
            worker_pool = [worker(wid) for wid in range(self.num_workers)]
            worker = mp_reader.multiprocess_reader(
                worker_pool, use_pipe=True, queue_size=1000)
            r = paddle.reader.buffered(worker, self.buf_size)

        for batch in r():
            yield batch

    def scan(self):
        """scan"""
        for example in self.dataset:
            yield example


if __name__ == "__main__":
    from base_dataset import BaseDataset, Subset
    dataset = GraphPropPredDataset(name="ogbg-molhiv")
    splitted_index = dataset.get_idx_split()
    train_dataset = Subset(dataset, splitted_index['train'])
    valid_dataset = Subset(dataset, splitted_index['valid'])
    test_dataset = Subset(dataset, splitted_index['test'])
    log.info("Train Examples: %s" % len(train_dataset))
    log.info("Val Examples: %s" % len(valid_dataset))
    log.info("Test Examples: %s" % len(test_dataset))

    #  train_loader = GraphDataloader(train_dataset, batch_size=3)
    #  for batch_data in train_loader:
    #      graphs, labels = batch_data
    #      print(labels.shape)
    #      time.sleep(4)
