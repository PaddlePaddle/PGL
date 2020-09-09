#-*- coding: utf-8 -*-
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
"""dataset
"""


class Dataset(object):
    """An abstract class represening Dataset.
    Generally, all datasets should subclass it.
    All subclasses should overwrite :code:`__getitem__` and :code:`__len__`.
    """

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class StreamDataset(object):
    """An abstract class represening StreamDataset which has unknown length.
    Generally, all unknown length datasets should subclass it.
    All subclasses should overwrite :code:`__iter__`.
    """

    def __iter__(self):
        raise NotImplementedError
