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
"""This package implements common layers to help building pooling operators.
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import paddle.fluid as F
import paddle.fluid.layers as L

import pgl

__all__ = ['Set2Set']


class Set2Set(object):
    """Implementation of set2set pooling operator.

    This is an implementation of the paper ORDER MATTERS: SEQUENCE TO SEQUENCE 
    FOR SETS (https://arxiv.org/pdf/1511.06391.pdf).
    """

    def __init__(self, input_dim, n_iters, n_layers):
        """
        Args:
            input_dim: hidden size of input data.
            n_iters: number of set2set iterations.
            n_layers: number of lstm layers.
        """
        self.input_dim = input_dim
        self.output_dim = 2 * input_dim
        self.n_iters = n_iters

        # this's set2set n_layers, lstm n_layers = 1
        self.n_layers = n_layers

    def forward(self, feat):
        """
        Args:
            feat: input feature with shape [batch, n_edges, dim].
        
        Return:
            output_feat: output feature of set2set pooling with shape [batch, 2*dim].
        """

        seqlen = 1
        h = L.fill_constant_batch_size_like(
            feat, [1, self.n_layers, self.input_dim], "float32", 0)
        h = L.transpose(h, [1, 0, 2])
        c = h

        # [seqlen, batch, dim]
        q_star = L.fill_constant_batch_size_like(
            feat, [1, seqlen, self.output_dim], "float32", 0)
        q_star = L.transpose(q_star, [1, 0, 2])

        for _ in range(self.n_iters):

            # q [seqlen, batch, dim]
            # h [layer, batch, dim]
            q, h, c = L.lstm(
                q_star,
                h,
                c,
                seqlen,
                self.input_dim,
                self.n_layers,
                is_bidirec=False)

            # e [batch, seqlen, n_edges]
            e = L.matmul(L.transpose(q, [1, 0, 2]), feat, transpose_y=True)
            # alpha [batch, seqlen, n_edges]
            alpha = L.softmax(e)

            # readout [batch, seqlen, dim]
            readout = L.matmul(alpha, feat)
            readout = L.transpose(readout, [1, 0, 2])

            # q_star [seqlen, batch, dim + dim]
            q_star = L.concat([q, readout], -1)

        return L.squeeze(q_star, [0])
