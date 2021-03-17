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
"""sampler
"""

import time
import numpy as np


class Sampler(object):
    """Sampler
    """

    def __init__(self, dataset, batch_size=1, drop_last=False, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __iter__(self):
        perm = np.arange(0, len(self.dataset))
        if self.shuffle:
            np.random.shuffle(perm)

        batch = []
        for idx in perm:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        length = len(self.dataset)
        if self.drop_last:
            length = length // self.batch_size
        else:
            length = (length + self.batch_size - 1) // self.batch_size
        return length


class StreamSampler(object):
    """StreamSampler
    """

    def __init__(self, dataset, batch_size=1, drop_last=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = [i for i in range(self.batch_size)]
        while True:
            yield batch
