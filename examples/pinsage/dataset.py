# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os

import paddle
import numpy as np
from paddle.io import get_worker_info

from pgl import graph_kernel
from pgl.utils.logger import log
from pgl.sampling import pinsage_sample
from pgl.utils.data import Dataset


def batch_fn(batch_ex, graph, samples, top_k=50):
    batch_train_samples = []
    batch_train_labels = []
    for i, l in batch_ex:
        batch_train_samples.append(i)
        batch_train_labels.append(l)

    subgraphs = pinsage_sample(
        graph, batch_train_samples, samples, top_k=top_k)
    subgraph, sample_index, node_index = subgraphs[0]

    node_label = np.array(batch_train_labels, dtype="int64").reshape([-1, 1])

    return subgraph, sample_index, node_index, node_label


class ShardedDataset(Dataset):
    def __init__(self, data_index, data_label, mode="train"):
        worker_info = get_worker_info()
        if worker_info is None or mode != "train":
            self.data = [data_index, data_label]
        else:
            self.data = [
                data_index[worker_info.id::worker_info.num_workers],
                data_label[worker_info.id::worker_info.num_workers]
            ]

    def __getitem__(self, idx):
        return [data[idx] for data in self.data]

    def __len__(self):
        return len(self.data[0])
