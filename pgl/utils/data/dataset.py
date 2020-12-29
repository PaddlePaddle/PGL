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
"""dataset
"""

import os
import sys
import numpy as np
import json
import io
from subprocess import Popen, PIPE


class HadoopUtil(object):
    """Implementation of some common hadoop operations.
    """

    def __init__(self, hadoop_bin, fs_name, fs_ugi):
        self.hadoop_bin = hadoop_bin
        self.fs_name = fs_name
        self.fs_ugi = fs_ugi

    def ls(self, path):
        """ hdfs_ls """
        cmd = self.hadoop_bin + " fs -D fs.default.name=" + self.fs_name
        cmd += " -D hadoop.job.ugi=" + self.fs_ugi
        cmd += " -ls " + path
        cmd += " | awk '{print $8}'"
        with os.popen(cmd) as reader:
            filelist = reader.read().split()
        return filelist

    def open(self, filename, encoding='utf-8'):
        """ hdfs_file_open """
        cmd = self.hadoop_bin + " fs -D fs.default.name=" + self.fs_name
        cmd += " -D hadoop.job.ugi=" + self.fs_ugi
        cmd += " -cat " + filename

        p = Popen(cmd, shell=True, stdout=PIPE)
        p = io.TextIOWrapper(p.stdout, encoding=encoding, errors='ignore')
        return p


class Dataset(object):
    """An abstract class represening Dataset.
    Generally, all datasets should subclass it.
    All subclasses should overwrite :code:`__getitem__` and :code:`__len__`.

    Examples:
        .. code-block:: python

            from pgl.utils.data import Dataset

            class MyDataset(Dataset):
                def __init__(self):
                    self.data = list(range(0, 40))

                def __getitem__(self, idx):
                    return self.data[idx]

                def __len__(self):
                    return len(self.data)
    """

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class StreamDataset(object):
    """An abstract class represening StreamDataset which has unknown length.
    Generally, all unknown length datasets should subclass it.
    All subclasses should overwrite :code:`__iter__`.

    Examples:
        .. code-block:: python

            from pgl.utils.data import StreamDataset

            class MyStreamDataset(StreamDataset):
                def __init__(self):
                    self.data = list(range(0, 40))
                    self.count = 0

                def __iter__(self):
                     for data in self.dataset:
                        self.count += 1
                        if self.count % self._worker_info.num_workers != self._worker_info.fid:
                            continue
                        # do something (like parse data)  of your data
                        time.sleep(0.1)
                        yield data
    """

    def __iter__(self):
        raise NotImplementedError

    def _set_worker_info(self, worker_info):
        self._worker_info = worker_info


class HadoopDataset(StreamDataset):
    """An abstract class represening HadoopDataset which loads data from hdfs.
    All subclasses should overwrite :code:`__iter__`.

    Examples:
        .. code-block:: python

            from pgl.utils.data import HadoopDataset

            class MyHadoopDataset(HadoopDataset):
                def __init__(self, data_path, hadoop_bin, fs_name, fs_ugi):
                    super(MyHadoopDataset, self).__init__(hadoop_bin, fs_name, fs_ugi)
                    self.data_path = data_path

                def __iter__(self):
                    for line in self._line_data_generator():
                        yield line    

                def _line_data_generator(self):
                    paths = self.hadoop_util.ls(self.data_path)
                    paths = sorted(paths)
                    for idx, filename in enumerate(paths):
                        if idx % self._worker_info.num_workers != self._worker_info.fid:
                            continue
                        with self.hadoop_util.open(filename) as f:
                            for line in f:
                                yield line
    """

    def __init__(self, hadoop_bin, fs_name, fs_ugi):
        self.hadoop_util = HadoopUtil(
            hadoop_bin=hadoop_bin, fs_name=fs_name, fs_ugi=fs_ugi)

    def __iter__(self):
        raise NotImplementedError
