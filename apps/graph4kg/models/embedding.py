# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved
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

import os
import time

import paddle
import numpy as np
import paddle.distributed as dist

from utils.helper import uniform


class NumPyEmbedding(object):
    """NumPy Embedding in mmap mode

    Args:

        num_embeddings: the size of the dictionary of embeddings.

        embedding_dim: the dimension of embedding vectors.

        init_value (optional: float or a tuple of two floats): 
            the range of initialization.
            (-init_value, init_value) for single float.
        
        weight_path: the file to save and load weight.

        scale_type: the parameter for OTE model.
    """

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 low=1.,
                 high=None,
                 weight_path='./__np_embedding.npy',
                 optimizer='AdaGrad',
                 learning_rate=1e-3):
        super(NumPyEmbedding, self).__init__()
        self._weight_path = weight_path
        self._moment_path = os.path.join(
            os.path.dirname(self._weight_path),
            os.path.basename(self._weight_path).strip('.npy') + '__moment.npy')
        self._init_weight(num_embeddings, embedding_dim, low, high)

        self.trace = []
        self._stop_gradient = False

        self._optim_mode = optimizer.lower()
        self._set_optimizer()
        self._lr = learning_rate

        self._async_q = None
        self._async_p = None

    def __call__(self, index):
        if isinstance(index, paddle.Tensor):
            index = index.numpy()
        tensors = paddle.to_tensor(self.weight[index])
        tensors.stop_gradient = self._stop_gradient
        if not self._stop_gradient:
            self.trace.append((index, tensors))
        return tensors

    def eval(self):
        """For evaluation without gradients
        """
        self.trace = []
        self._stop_gradient = True

    def train(self):
        """Fro training with gradient trace
        """
        self._stop_gradient = False

    @property
    def curr_emb(self):
        """Return current embeddings
        """
        data = [x for _, x in self.trace]
        return paddle.concat(data, axis=0)

    def get(self, index):
        """Get embeddings of index
        """
        assert isinstance(index, np.ndarray)
        return self.weight[index]

    def step(self):
        """Update embeddings according to trace
        """
        with paddle.no_grad():
            for index, tensors in self.trace:
                grad = tensors.grad.numpy()
                self._update(grad, index)
        self.trace = []

    def _set_optimizer(self):
        if self._optim_mode == 'adagrad':
            self._update = self._update_adagrad
            self._init_moment()
        elif self._optim_mode == 'sgd':
            self._update = self._update_sgd
        else:
            raise ValueError('update method %s is not supported!' %
                             self._optim_mode)

    def _init_weight(self, num_embeddings, embedding_dim, low, high):
        if dist.get_rank() == 0:
            if high is None:
                high = low
                low = -low
            assert low < high, 'invalid initialization range!'
            weight = uniform(low, high, (num_embeddings, embedding_dim))
            np.save(self._weight_path, weight)
            del weight
        else:
            while True:
                if os.path.exists(self._weight_path):
                    break
                time.sleep(5)
        self.weight = np.load(self._weight_path, mmap_mode='r+')

    def _init_moment(self):
        if dist.get_rank() == 0:
            moment = np.zeros_like(self.weight, dtype=np.float32)
            np.save(self._moment_path, moment)
            del moment
        else:
            while True:
                if os.path.exists(self._moment_path):
                    break
                time.sleep(5)
        self._moment = np.load(self._moment_path, mmap_mode='r+')

    def _update_adagrad(self, grad, index):
        grad_square = grad * grad
        self._moment[index] += grad_square
        std = np.sqrt(self._moment[index]) + 1e-6
        grad = -self._lr * grad / std
        self.weight[index] += grad

    def _update_sgd(self, grad, index):
        grad = -self._lr * grad
        self.weight[index] += grad
