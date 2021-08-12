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
"""dataloader
"""
import warnings
import time
import numpy as np
from collections import namedtuple

import paddle

from pgl.utils import mp_reader
from pgl.utils.data.dataset import Dataset, StreamDataset
from pgl.utils.data.sampler import Sampler, StreamSampler

WorkerInfo = namedtuple("WorkerInfo", ["num_workers", "fid"])


class Dataloader(object):
    """Dataloader for loading batch data

    Example:

        .. code-block:: python

            from pgl.utils.data import Dataset
            from pgl.utils.data.dataloader import Dataloader

            class MyDataset(Dataset):
                def __init__(self):
                    self.data = list(range(0, 40))

                def __getitem__(self, idx):
                    return self.data[idx]

                def __len__(self):
                    return len(self.data)

            def collate_fn(batch_examples):
                inputs = np.array(batch_examples, dtype="int64")
                return inputs

            dataset = MyDataset()
            loader = Dataloader(dataset, 
                                batch_size=3,
                                drop_last=False,
                                shuffle=True,
                                num_workers=4,
                                collate_fn=collate_fn)

            for batch_data in loader:
                print(batch_data)

    """

    def __init__(self,
                 dataset,
                 batch_size=1,
                 drop_last=False,
                 shuffle=False,
                 num_workers=1,
                 collate_fn=None,
                 buf_size=1000,
                 stream_shuffle_size=0):

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.buf_size = buf_size
        self.drop_last = drop_last
        self.stream_shuffle_size = stream_shuffle_size

        if self.shuffle and isinstance(self.dataset, StreamDataset):
            warn_msg = "The argument [shuffle] should not be True with StreamDataset. " \
                    "It will be ignored. " \
                    "You might want to set [stream_shuffle_size] with StreamDataset."
            warnings.warn(warn_msg)

        if self.stream_shuffle_size > 0 and self.batch_size > stream_shuffle_size:
            warn_msg = "stream_shuffle_size should be larger than batch_size, "
            warn_msg += "but got [stream_shuffle_size=%s] " % (
                self.stream_shuffle_size)
            warn_msg += "smaller than [batch_size=%s]. " % (self.batch_size)
            warn_msg += "stream_shuffle_size will be set to %s." % (
                self.batch_size)
            warnings.warn(warn_msg)

        if self.stream_shuffle_size > 0 and isinstance(self.dataset, Dataset):
            warn_msg = "[stream_shuffle_size] should not be set with Dataset. " \
                    "It will be ignored. " \
                    "You might want to set [shuffle] with Dataset."
            warnings.warn(warn_msg)

        if self.num_workers < 1:
            raise ValueError("num_workers(default: 1) should be larger than 0, " \
                        "but got [num_workers=%s] < 1." % self.num_workers)

        if isinstance(self.dataset, StreamDataset):  # for stream data
            # generating a iterable sequence for produce batch data without repetition
            self.sampler = StreamSampler(
                self.dataset,
                batch_size=self.batch_size,
                drop_last=self.drop_last)
        else:
            self.sampler = Sampler(
                self.dataset,
                batch_size=self.batch_size,
                drop_last=self.drop_last,
                shuffle=self.shuffle)

    def __len__(self):
        if not isinstance(self.dataset, StreamDataset):
            return len(self.sampler)
        else:
            raise "StreamDataset has no length"

    def __iter__(self):
        # random seed will be fixed when using multiprocess, 
        # so set seed explicitly every time
        np.random.seed()
        if self.num_workers == 1:
            workers = _DataLoaderIter(self, 0)
        else:
            worker_pool = [
                _DataLoaderIter(self, wid) for wid in range(self.num_workers)
            ]
            workers = mp_reader.multiprocess_reader(worker_pool)

        for batch in workers():
            yield batch

    def __call__(self):
        return self.__iter__()


class _DataLoaderIter(object):
    """Iterable DataLoader Object
    """

    def __init__(self, dataloader, fid=0):
        self.dataset = dataloader.dataset
        self.sampler = dataloader.sampler
        self.collate_fn = dataloader.collate_fn
        self.num_workers = dataloader.num_workers
        self.drop_last = dataloader.drop_last
        self.batch_size = dataloader.batch_size
        self.stream_shuffle_size = dataloader.stream_shuffle_size
        self.fid = fid

    def _data_generator(self):
        for count, indices in enumerate(self.sampler):

            if count % self.num_workers != self.fid:
                continue

            batch_data = [self.dataset[i] for i in indices]

            if self.collate_fn is not None:
                yield self.collate_fn(batch_data)
            else:
                yield batch_data

    def _streamdata_generator(self):
        self._worker_info = WorkerInfo(
            num_workers=self.num_workers, fid=self.fid)
        self.dataset._set_worker_info(self._worker_info)

        dataset = iter(self.dataset)
        for indices in self.sampler:
            batch_data = []
            for _ in indices:
                try:
                    batch_data.append(next(dataset))
                except StopIteration:
                    break

            if len(batch_data) == 0 or (self.drop_last and
                                        len(batch_data) < len(indices)):
                break
                #  raise StopIteration

            if self.collate_fn is not None:
                yield self.collate_fn(batch_data)
            else:
                yield batch_data

    def _stream_shuffle_data_generator(self):
        def _batch_stream_data_generator():
            dataset = iter(self.dataset)
            batch_data = []
            while True:
                try:
                    batch_data.append(next(dataset))
                except StopIteration:
                    break

                if len(batch_data) == self.batch_size:
                    yield batch_data
                    batch_data = []

            if not self.drop_last and len(batch_data) > 0:
                yield batch_data
                batch_data = []

        def _batch_stream_shuffle_generator():
            buffer_list = []
            batch_data = []
            for examples in _batch_stream_data_generator():
                if len(buffer_list) < self.stream_shuffle_size:
                    buffer_list.extend(examples)
                else:
                    rand_idx = np.random.randint(0,
                                                 len(buffer_list),
                                                 len(examples))
                    for idx, e in zip(rand_idx, examples):
                        batch_data.append(buffer_list[idx])
                        buffer_list[idx] = e

                    yield batch_data
                    batch_data = []

            if len(buffer_list) > 0:
                np.random.shuffle(buffer_list)
                batch_data = []
                for e in buffer_list:
                    batch_data.append(e)
                    if len(batch_data) == self.batch_size:
                        yield batch_data
                        batch_data = []

                if not self.drop_last and len(batch_data) > 0:
                    yield batch_data
                    batch_data = []

        self._worker_info = WorkerInfo(
            num_workers=self.num_workers, fid=self.fid)
        self.dataset._set_worker_info(self._worker_info)

        for batch_data in _batch_stream_shuffle_generator():
            if self.collate_fn is not None:
                yield self.collate_fn(batch_data)
            else:
                yield batch_data

    def __iter__(self):
        if isinstance(self.dataset, StreamDataset):
            if self.stream_shuffle_size > 0:
                data_generator = self._stream_shuffle_data_generator
            else:
                data_generator = self._streamdata_generator
        else:
            data_generator = self._data_generator

        for batch_data in data_generator():
            yield batch_data

    def __call__(self):
        return self.__iter__()
