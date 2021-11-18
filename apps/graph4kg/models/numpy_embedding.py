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
import multiprocessing as mp

from utils import uniform, thread_wrapper, timer_wrapper


class NumPyEmbedding(object):
    """NumPy Embedding in mmap mode

    Args:

        num_embeddings: the size of the dictionary of embeddings.

        embedding_dim: the dimension of embedding vectors.

        init_value (optional: float or a tuple of two floats): 
            the range of initialization.
            (-init_value, init_value) for single float.

        weight_path: the file to save and load weight.

    """

    def __init__(self,
                 num_embeddings=1,
                 embedding_dim=1,
                 high=1,
                 low=None,
                 weight=None,
                 weight_path='./__np_embedding.npy',
                 optimizer='AdaGrad',
                 learning_rate=1e-3):
        super(NumPyEmbedding, self).__init__()
        self._num_embed = num_embeddings
        self._embed_dim = embedding_dim
        self._low = -high if low is None else low
        self._high = high
        self._weight_path = weight_path
        self._moment_path = os.path.join(
            os.path.dirname(self._weight_path),
            os.path.basename(self._weight_path).strip('.npy') + '_moment.npy')
        self._init_weight(weight)

        self.trace = []
        self._stop_gradient = False

        self._optim_mode = optimizer.lower()
        self._lr = learning_rate
        self._update = self._set_optimizer()

        self._process_worker = 4
        self._async_q = None
        self._async_p = []
        for i in range(self._process_worker):
            self._async_p.append(None)

    def __call__(self, index):
        if isinstance(index, paddle.Tensor):
            index = index.numpy()
        tensors = paddle.to_tensor(self.weight[index])
        tensors.stop_gradient = self._stop_gradient
        if not self._stop_gradient:
            self.trace.append((index, tensors))
        return tensors

    @classmethod
    def from_weight(cls, weight, weight_path, optimizer, learning_rate):
        """Initializa NumPyEmbedding with a pre-defined array
        """
        return cls(weight=weight, weight_path=weight_path, \
            optimizer=optimizer, learning_rate=learning_rate)

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
    def weight_path(self):
        """Return the path of mmap embeddings
        """
        return self._weight_path

    @property
    def curr_emb(self):
        """Return current embeddings
        """
        if len(self.trace) == 0:
            return None
        data = [x for _, x in self.trace]
        return paddle.concat(data, axis=0)

    def get(self, index):
        """Get embeddings of index
        """
        assert isinstance(index, np.ndarray)
        return self.weight[index]

    def start_async_update(self):
        """initialize the async update
        """
        self._async_q = mp.Queue(self._process_worker * 100)
        for i in range(self._process_worker):
            self._async_p[i] = mp.Process(
                target=self.async_update, args=(self._async_q, ))

        for i in range(self._process_worker):
            self._async_p[i].start()

    def finish_async_update(self):
        """Notify the async update process to quit
        """
        for i in range(self._process_worker):
            self._async_q.put((None, None))

        for i in range(self._process_worker):
            self._async_p[i].join()

    def step(self):
        """Update embeddings according to self.trace
        """
        with paddle.no_grad():
            for index, tensors in self.trace:
                grad_trace = self.create_trace(tensors)
                if self._async_q is not None:
                    self._async_q.put([index, grad_trace])
                else:
                    self._update(index, grad_trace)
        self.trace = []

    def step_trace(self, trace):
        """Update embeddings according to given trace
        """
        with paddle.no_grad():
            index, grad_trace = trace
            if self._async_q is not None:
                self._async_q.put([index, grad_trace])
            else:
                self._update(index, grad_trace)

    def create_trace(self, index, embeds):
        """Create gradient trace for given paddle.tensor
        """
        if embeds is None:
            return None
        index = index.numpy()
        if self._optim_mode == 'adagrad':
            grad = embeds.grad
            grad_square = (grad * grad).mean(axis=-1).numpy()
            grads = [grad.numpy(), grad_square]
        elif self._optim_mode == 'sgd':
            grads = [embeds.grad.numpy()]
        return [index, grads]

    @thread_wrapper
    def async_update(self, queue):
        """Update embeddings asynchronously
        """
        while True:
            (grad_index, grad_trace) = queue.get()
            if grad_index is None:
                return
            with paddle.no_grad():
                self._update(grad_index, grad_trace)

    def _set_optimizer(self):
        if self._optim_mode == 'adagrad':
            self._init_moment()
            return self._update_adagrad
        elif self._optim_mode == 'sgd':
            return self._update_sgd
        else:
            raise ValueError('update method %s is not supported!' %
                             self._optim_mode)

    def _init_weight(self, weight):
        if dist.get_rank() == 0:
            if weight is None:
                assert self._low < self._high, 'invalid initialization range!'
                embed_shape = (self._num_embed, self._embed_dim)
                weight = uniform(self._low, self._high, embed_shape)
            np.save(self._weight_path, weight)
            del weight
        else:
            while True:
                if os.path.exists(self._weight_path):
                    break
                time.sleep(5)
        self.weight = np.load(
            self._weight_path, allow_pickle=True, mmap_mode='r+')

    def _init_moment(self):
        if dist.get_rank() == 0:
            moment = np.zeros((self.weight.shape[0], ), dtype=np.float32)
            np.save(self._moment_path, moment)
            del moment
        else:
            while True:
                if os.path.exists(self._moment_path):
                    break
                time.sleep(5)
        self._moment = np.load(self._moment_path, mmap_mode='r+')

    def _update_adagrad(self, index, grad_trace):
        grad, grad_square = grad_trace
        self._moment[index] += grad_square
        std = np.sqrt(self._moment[index]) + 1e-10
        grad = -self._lr * grad / std.reshape((-1, 1))
        self.weight[index] += grad

    def _update_sgd(self, index, grad):
        grad = -self._lr * grad
        self.weight[index] += grad
