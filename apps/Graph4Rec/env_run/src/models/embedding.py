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
"""Model Definition
"""
import math
import time

import paddle
import paddle.nn as nn

__all__ = ["BaseEmbedding", "SparseEmbedding"]


class BaseEmbedding(nn.Layer):
    def __init__(self,
                 num,
                 dim,
                 sparse=None,
                 weight_attr=None,
                 name=None,
                 **kwargs):
        super(BaseEmbedding, self).__init__()
        self.embedding = nn.Embedding(
            num, dim, sparse=sparse, weight_attr=weight_attr, name=name)

    def forward(self, x):
        return self.embedding(x)


class SparseEmbedding(object):
    def __init__(self,
                 num,
                 dim,
                 sparse=None,
                 weight_attr=None,
                 name=None,
                 **kwargs):
        self.num = num
        self.dim = dim
        self.sparse = sparse
        self.emb_attr = weight_attr
        self.name = name

    def forward(self, x):

        x_emb = paddle.static.nn.sparse_embedding(
            x, [self.num, self.dim], param_attr=self.emb_attr)

        return x_emb

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
