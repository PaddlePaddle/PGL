# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2021, rusty1s(github).
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
"""
    History storage module for GNNAutoScale, refers to `pygas`
    (https://github.com/rusty1s/pyg_autoscale).
"""

import numpy as np
import paddle
from pgl.utils.logger import log


class History(paddle.nn.Layer):
    """History storage module of GNNAutoScale.

    Args:
        
        num_embs (int): Usually the same with number of nodes in a graph.

        emb_dims (int): Should be set as the hidden size of gnn models. 

    """

    def __init__(self, num_embs, emb_dim):
        super().__init__()

        self.num_embs = num_embs
        self.emb_dim = emb_dim

        numpy_data = np.zeros((self.num_embs, self.emb_dim), dtype="float32")
        self.emb = paddle.to_tensor(numpy_data, place=paddle.CUDAPinnedPlace())

    def forward(self, *args, **kwargs):
        raise NotImplementedError
