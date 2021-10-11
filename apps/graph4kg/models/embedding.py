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

from utils.helper import uniform, thread_wrapper, async_update

# from models.shared_numpy import SharedArray
# from pgl.utils.mp_reader import serialize_data


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
                 num_embeddings=1,
                 embedding_dim=1,
                 low=1.,
                 high=None,
                 weight_path='./__np_embedding.npy',
                 optimizer='AdaGrad',
                 learning_rate=1e-3,
                 load_mode=False):
        super(NumPyEmbedding, self).__init__()
        self._load_mode = load_mode
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
        self._async_q = mp.Queue(1)
        self._async_p = mp.Process(
            target=async_update, args=(self, self._async_q))
        self._async_p.start()

    def finish_async_update(self):
        """Notify the async update process to quit
        """
        self._async_q.put((None, None))
        self._async_p.join()

    def step(self):
        """Update embeddings according to self.trace
        """
        with paddle.no_grad():
            for index, tensors in self.trace:
                grad = tensors.grad.numpy()
                if self._async_q is not None:
                    # index = SharedArray.copy_from(index)
                    # grad = SharedArray.copy_from(grad)
                    # self._async_q.put(serialize_data([index, grad]))
                    self._async_q.put([index, grad])
                else:
                    self._update(grad, index)
        self.trace = []

    def step_trace(self, trace):
        """Update embeddings according to given trace
        """
        with paddle.no_grad():
            index, grad = trace
            if self._async_q is not None:
                # index.detach()
                # grad.detach()
                # index = index.cpu()._share_memory()
                # grad = grad.cpu()._share_memory()
                # grad_shape = grad.shape
                # index = SharedArray.copy_from(index)
                # grad = SharedArray.copy_from(grad.reshape((-1,)))
                # self._async_q.put([index, grad, grad_shape])
                self._async_q.put([index, grad])
                # paddle.fluid.core._remove_tensor_list_mmap_fds([index, grad])
            else:
                self._update(grad, index)

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
        if dist.get_rank() == 0 and not self._load_mode:
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
        if False:
            self.weight = np.load(self._weight_path, mmap_mode='r')
        else:
            self.weight = np.load(self._weight_path, mmap_mode='r+')

    def _init_moment(self):
        if dist.get_rank() == 0:
            moment = np.zeros(self.weight.shape[0], dtype=np.float32)
            np.save(self._moment_path, moment)
            del moment
        else:
            while True:
                if os.path.exists(self._moment_path):
                    break
                time.sleep(5)
        self._moment = np.load(self._moment_path, mmap_mode='r+')

    def _update_adagrad(self, grad, index):
        grad_square = (grad * grad).mean(1)
        self._moment[index] += grad_square
        std = np.sqrt(self._moment[index]) + 1e-6
        grad = -self._lr * grad / std.reshape((-1, 1))
        self.weight[index] += grad

    def _update_sgd(self, grad, index):
        grad = -self._lr * grad
        self.weight[index] += grad
