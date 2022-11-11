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

import numpy as np
import paddle
import paddle.distributed as dist


def set_current_device_id():
    """ The except place may not consist with cuda current device id.
        we reset is in here.
    """
    import paddle
    curr_dev = paddle.device.get_device()
    select_gpu = os.getenv("FLAGS_selected_gpus", "0")
    paddle.set_flags({
        'FLAGS_selected_gpus': os.getenv("FLAGS_selected_gpus", "0")
    })
    if "gpu" in curr_dev and select_gpu != curr_dev.split(":")[-1]:
        paddle.set_device("gpu:" + select_gpu)

    curr_dev_id = paddle.framework.core.get_cuda_current_device_id()
    if "gpu" in curr_dev and select_gpu != str(curr_dev_id):
        paddle.zeros([])


def uniform(low, high, size, dtype=np.float32):
    """Memory efficient uniform implementation.
    """
    rng = np.random.default_rng(0)
    out = (high - low) * rng.random(size, dtype=dtype) + low
    return out


def async_update(self, queue, event):
    """Update embeddings asynchronously
    """
    set_current_device_id()
    weight = None
    _moment = None

    while True:
        (index, grad_trace) = queue.get()
        if index is None:
            return
        if weight is None:
            # Must reload in here, the mmap obj can't pass with spawn.
            weight = np.load(
                self._weight_path, allow_pickle=True, mmap_mode='r+')
            _moment = np.load(self._moment_path, mmap_mode='r+')

        index = index.clone()
        grad_trace = [x.clone() for x in grad_trace]
        event.set()  # Release the main thread.

        # Copy tensor to numpy.
        grad, grad_square = grad_trace
        index = index.numpy()
        grad = grad.numpy()
        grad_square = grad_square.numpy()

        # update adagrad
        _moment[index] += grad_square
        std = np.sqrt(_moment[index]) + 1e-10
        grad = -self._lr * grad / std.reshape((-1, 1))
        weight[index] += grad


class SharedEmbedding(object):
    """
    SharedEmbedding in mmap mode.

    Args:
        num_embeddings (int):
            The number of embeddings.
        embedding_dim (int):
            The dimension of embeddings.
        high (float):
            Optional. The upper bound of embedding values.
        low (float):
            Optional. The lower bound of embedding values.
        weight (np.ndarray):
            Optional. The array used to initialize embeddings.
        weight_path (str):
            The file to save and load embeddings.
        optimizer (str):
            The optimizer used to update embeddings during training. Choices: sgd, adagrad.
        learning_rate (float):
            The learning rate of optimizer.
        init_mode (str):
            The source of embeddings initialization. Choices: range, array, file.
        num_workers (int):
            The number of processes to update gradients.
    """

    def __init__(self,
                 num_embeddings=1,
                 embedding_dim=1,
                 high=1,
                 low=None,
                 weight=None,
                 weight_path='./__default_shared_embedding.npy',
                 optimizer='adagrad',
                 learning_rate=0.1,
                 init_mode='range',
                 num_workers=1):
        super(SharedEmbedding, self).__init__()
        self._num_embed = num_embeddings
        self._embed_dim = embedding_dim
        self._init_mode = init_mode
        self._low = low
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

        self._process_worker = num_workers
        self._async_q = None
        self._async_e = None
        self._async_p = []
        for i in range(self._process_worker):
            self._async_p.append(None)

    def __call__(self, index):
        if isinstance(index, paddle.Tensor):
            index = index.numpy()
        tensors = paddle.to_tensor(self.weight[index])
        tensors.stop_gradient = self._stop_gradient
        index = paddle.to_tensor(index)
        if not self._stop_gradient:
            self.trace.append((index, tensors))
        return tensors

    @classmethod
    def from_array(cls,
                   weight,
                   weight_path,
                   optimizer='AdaGrad',
                   learning_rate=0.1,
                   num_workers=1):
        """Initialize SharedEmbedding with a pre-defined array
        """
        return cls(weight=weight, weight_path=weight_path, \
            optimizer=optimizer, learning_rate=learning_rate, init_mode='array', num_workers=num_workers)

    @classmethod
    def from_file(cls,
                  weight_path,
                  optimizer='AdaGrad',
                  learning_rate=0.1,
                  num_workers=1):
        """Initialize SharedEmbedding from array stored in weight_path
        """
        return cls(weight_path=weight_path, \
            optimizer=optimizer, learning_rate=learning_rate, init_mode='file', num_workers=num_workers)

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

        #import paddle.incubate.multiprocessing as mp
        def device_id_hook(*args, **kwargs):
            set_current_device_id()
            return paddle.incubate.multiprocessing.reductions.reduce_tensor(
                *args, **kwargs)

        import paddle.incubate.multiprocessing as mp
        mp = mp.get_context("spawn")
        import multiprocessing
        multiprocessing.reduction.ForkingPickler._extra_reducers[
            paddle.Tensor] = device_id_hook

        self._async_q = mp.Queue(1)
        self._async_e = mp.Event()
        for i in range(self._process_worker):
            self._async_p[i] = mp.Process(
                target=async_update, args=(self, self._async_q, self._async_e))
            self._async_p[i].daemon = False
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
                grad_trace = self.create_trace(index, tensors)
                if self._async_q is not None:
                    self._async_q.put(grad_trace)
                    self._async_e.wait()
                    self._async_e.clear()
                else:
                    index, grads = grad_trace
                    self._update(index, grads)
        self.trace = []

    def step_trace(self, trace):
        """Update embeddings according to given trace
        """
        with paddle.no_grad():
            index, grad_trace = trace
            if self._async_q is not None:
                self._async_q.put([index, grad_trace], )
                self._async_e.wait()
                self._async_e.clear()
            else:
                self._update(index, grad_trace)

    def create_trace(self, index, embeds):
        """Create gradient trace for given paddle.tensor
        """
        if embeds is None:
            return None
        with paddle.no_grad():
            if self._optim_mode == 'adagrad':
                grad = embeds.grad
                grad_square = (grad * grad).mean(axis=-1)
                grads = [grad.detach(), grad_square.detach()]
            elif self._optim_mode == 'sgd':
                grads = [embeds.grad.detach()]
        return [index, grads]

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
            if self._init_mode == 'range':
                self._low = -self._high if self._low is None else self._low
                assert self._low < self._high, 'Invalid range to initialize SharedEmbedding!'
                embed_shape = (self._num_embed, self._embed_dim)
                weight = uniform(self._low, self._high, embed_shape)
                np.save(self._weight_path, weight)
                del weight
            elif self._init_mode == 'array':
                assert isinstance(
                    weight, np.ndarray
                ), 'Invalid weight type to initialize SharedEmbedding!'
                np.save(self._weight_path, weight)
                del weight
            elif self._init_mode != 'file':
                raise ValueError(
                    'Initialization mode {} not supportted in SharedEmbedding'.
                    format(self._init_mode))
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
        index = index.numpy()
        grad, grad_square = grad_trace
        grad = grad.numpy()
        grad_square = grad_square.numpy()

        self._moment[index] += grad_square
        std = np.sqrt(self._moment[index]) + 1e-10
        grad = -self._lr * grad / std.reshape((-1, 1))
        self.weight[index] += grad

    def _update_sgd(self, index, grad):
        grad = -self._lr * grad
        self.weight[index] += grad
