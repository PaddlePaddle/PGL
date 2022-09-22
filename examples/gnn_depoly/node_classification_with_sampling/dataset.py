# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
from paddle.io import Dataset, DataLoader
from pgl.sampling import graphsage_sample


class ShardedDataset(Dataset):
    def __init__(self, data_index, data_label):
        self.data_index = data_index
        self.data_label = data_label

    def __getitem__(self, idx):
        return self.data_index[idx], self.data_label[idx]

    def __len__(self):
        return len(self.data_index)


def generate_batch_infer_data(batch_ex, graph, samples):
    batch_test_samples = batch_ex[0]
    batch_test_labels = batch_ex[1]

    subgraphs = graphsage_sample(graph, batch_test_samples, samples)
    subgraph, sample_index, node_index = subgraphs[0]

    node_label = np.array(batch_test_labels, dtype="int64")
    return subgraph, sample_index, node_index, node_label
