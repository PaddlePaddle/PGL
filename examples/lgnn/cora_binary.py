# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

# Cora Binary dataset
import os
import sys

import numpy as np

import pgl
from pgl.utils.data import Dataset


class CoraBinary(Dataset):
    """A mini-dataset for binary classification task using Cora.
    """

    def __init__(self, raw_dir=None):
        super(CoraBinary, self).__init__()
        self.num = 21
        self.save_path = "./cora_binary"
        if not os.path.exists(self.save_path):
            os.system(
                "wget http://10.255.129.12:8122/cora_binary.zip && unzip cora_binary.zip"
            )
        self.graphs, self.line_graphs, self.labels = [], [], []
        self.load()

    def load(self):
        for idx in range(self.num):
            self.graphs.append(
                pgl.Graph.load(
                    os.path.join(self.save_path, str(idx), "graph")))
            self.line_graphs.append(
                pgl.Graph.load(
                    os.path.join(self.save_path, str(idx), "line_graph")))
            self.labels.append(
                np.load(os.path.join(self.save_path, str(idx), "labels.npy")))

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, i):
        return (self.graphs[i], self.line_graphs[i], self.labels[i])


if __name__ == "__main__":
    c = CoraBinary()
    for data in c:
        print(data)
