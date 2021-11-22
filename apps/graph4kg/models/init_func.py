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

import numpy as np
from numpy.random import RandomState

from utils import uniform, gram_schimidt_process


class InitFunction(object):
    """Initialization strategies.
    """

    def __init__(self, args):
        self.args = args

    def __call__(self, name, num_embed, embed_dim):
        self.num_embed = num_embed
        self.embed_dim = embed_dim

        if name == 'general_uniform':
            weight = self.general_uniform()
        elif name == 'quaternion_init':
            weight = self.quaternion_initialization()
        elif name == 'standard_uniform':
            weight = self.standard_uniform()
        elif name == 'ote_entity_uniform':
            weight = uniform(-0.01, 0.01, (self.num_embed, self.embed_dim))
        elif name == 'ote_scale_init':
            weight = self.standard_uniform()

            num_elem = self.args.ote_size
            use_scale = self.args.ote_scale > 0
            scale_init = 1. if self.args.ote_scale == 1 else 0.

            if use_scale:
                weight.reshape((-1, num_elem + 1))[:, -1] = scale_init
            weight_shape = weight.shape
            weight = weight.reshape([-1, num_elem, num_elem + int(use_scale)])
            weight = gram_schimidt_process(weight, num_elem, use_scale)
            weight = weight.reshape(weight_shape)
        else:
            raise ValueError(
                '{} initialization method not implemented!'.format(name))

        return weight

    def general_uniform(self):
        """General initialization method for most KE methods
        """
        embed_epsilon = 2.0
        init_value = (self.args.gamma + embed_epsilon) / self.embed_dim
        weight_shape = (self.num_embed, self.embed_dim)
        weight = uniform(-init_value, init_value, weight_shape)
        return weight

    def quaternion_initialization(self):
        """Initialization method for QuatE
        """
        sub_embed_dim = self.embed_dim // 4
        init_value = 1. / np.sqrt(2 * self.num_embed)
        rng = RandomState(123)

        weight_shape = (self.num_embed, sub_embed_dim)
        num_weights = np.prod(weight_shape)
        v_i = uniform(0., 1., num_weights)
        v_j = uniform(0., 1., num_weights)
        v_k = uniform(0., 1., num_weights)

        for i in range(num_weights):
            norm = np.sqrt(v_i[i]**2 + v_j[i]**2 + v_k[i]**2) + 1e-4
            v_i[i] /= norm
            v_j[i] /= norm
            v_k[i] /= norm
        v_i = v_i.reshape(weight_shape)
        v_j = v_j.reshape(weight_shape)
        v_k = v_k.reshape(weight_shape)

        modulus = uniform(-init_value, init_value, size=weight_shape, seed=123)
        phase = uniform(-np.pi, np.pi, size=weight_shape, seed=123)

        w_r = modulus * np.cos(phase)
        w_i = modulus * v_i * np.sin(phase)
        w_j = modulus * v_j * np.sin(phase)
        w_k = modulus * v_k * np.sin(phase)

        weight = np.concatenate(
            [w_r, w_i, w_j, w_k], axis=-1).astype(np.float32)
        return weight

    def standard_uniform(self):
        """Standard uniform
        """
        weight = uniform(0, 1, (self.num_embed, self.embed_dim))
        return weight
