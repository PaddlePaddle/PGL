# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved
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
""" gpu dataloader  """

import paddle
from paddle.common_ops_import import dygraph_only

__all__ = ["Dataloader_Cls"]


class Dataloader_Cls(object):
    """This dataloader is mainly for node classification task, and only supports single-gpu
       training currently. 
    """

    def __init__(self,
                 node_index,
                 node_label,
                 batch_size,
                 shuffle=True,
                 drop_last=False):

        if not isinstance(node_index, paddle.Tensor):
            node_index = paddle.to_tensor(node_index)
        if not isinstance(node_label, paddle.Tensor):
            node_label = paddle.to_tensor(node_label)

        self.node_index = node_index
        self.node_label = node_label
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    @paddle.no_grad()
    def __iter__(self):
        if self.shuffle:
            perm = paddle.randperm(
                self.node_index.shape[0], dtype=self.node_index.dtype)
        else:
            perm = paddle.arange(
                0, self.node_index.shape[0], dtype=self.node_index.dtype)

        start = 0
        while start < self.node_index.shape[0]:
            batch = perm[start:start + self.batch_size]
            if batch.shape[0] != self.batch_size and self.drop_last:
                break
            yield self.node_index[batch], self.node_label[batch]
            start += self.batch_size

    def __len__(self):
        return self.node_index.shape[0]
