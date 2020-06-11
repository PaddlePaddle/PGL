# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
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

import numpy as np
import collections
import paddle
import pgl
from pgl.utils.logger import log
from pgl.graph import Graph, MultiGraph

def batch_iter(data, batch_size):
    """node_batch_iter
    """
    size = len(data)
    perm = np.arange(size)
    np.random.shuffle(perm)
    start = 0
    while start < size:
        index = perm[start:start + batch_size]
        start += batch_size
        yield data[index]


def scan_batch_iter(data, batch_size):
    """scan_batch_iter
    """
    batch = []
    for example in data.scan():
        batch.append(example)
    if len(batch) == batch_size:
        yield batch
        batch = []

    if len(batch) > 0:
        yield batch


def label_to_onehot(labels):
    """Return one-hot representations of labels
    """
    onehot_labels = []
    for label in labels:
        if label == 0:
            onehot_labels.append([1, 0])
        else:
            onehot_labels.append([0, 1])
    onehot_labels = np.array(onehot_labels)
    return onehot_labels


class GraphDataloader(object):
    """Graph Dataloader
    """
    def __init__(self,
                dataset,
                graph_wrapper,
                batch_size,
                seed=0,
                buf_size=1000,
                shuffle=True):

        self.shuffle = shuffle
        self.seed = seed
        self.batch_size = batch_size
        self.dataset = dataset
        self.buf_size = buf_size
        self.graph_wrapper = graph_wrapper

    def batch_fn(self, batch_examples):
        """ batch_fun batch producer """
        graphs = [b[0] for b in batch_examples]
        labels = [b[1] for b in batch_examples]
        join_graph = MultiGraph(graphs)

        # normalize
        indegree = join_graph.indegree()
        norm = np.zeros_like(indegree, dtype="float32")
        norm[indegree > 0] = np.power(indegree[indegree > 0], -0.5)
        join_graph.node_feat["norm"] = np.expand_dims(norm, -1)
        
        feed_dict = self.graph_wrapper.to_feed(join_graph)
        labels = np.array(labels)
        feed_dict["labels_1dim"] = labels
        labels = label_to_onehot(labels)
        feed_dict["labels"] = labels

        graph_lod = join_graph.graph_lod
        graph_id = []
        for i in range(1, len(graph_lod)):
            graph_node_num = graph_lod[i] - graph_lod[i - 1]
            graph_id += [i - 1] * graph_node_num
        graph_id = np.array(graph_id, dtype="int32")
        feed_dict["graph_id"] = graph_id

        return feed_dict

    def batch_iter(self):
        """ batch_iter """
        if self.shuffle:
            for batch in batch_iter(self, self.batch_size):
                yield batch
        else:
            for batch in scan_batch_iter(self, self.batch_size):
                yield batch			

    def __len__(self):
        """__len__"""
        return len(self.dataset) 

    def __getitem__(self, idx):
        """__getitem__"""
        if isinstance(idx, collections.Iterable):
            return [self.dataset[bidx] for bidx in idx]
        else:
            return self.dataset[idx]

    def __iter__(self):
        """__iter__"""
        def func_run():
            for batch_examples in self.batch_iter():
                batch_dict = self.batch_fn(batch_examples)
                yield batch_dict

        r = paddle.reader.buffered(func_run, self.buf_size)

        for batch in r():
            yield batch

    def scan(self):
        """scan"""
        for example in self.dataset:
            yield example
