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

import unittest
import time
import multiprocessing as mp
import numpy as np
import paddle
import copy
from utils.helper import thread_wrapper, to_tensor
from models.shared_numpy import SharedArray
from models.embedding import NumPyEmbedding

paddle.ones([2, 2])
print("\n" * 10)


@thread_wrapper
def async_update(embeds, queue):
    while True:
        (grad_index, grad_value) = queue.get()
        if grad_index is None:
            return
        else:
            if type(grad_index) == SharedArray:
                index = grad_index.array
                value = grad_value.array.reshape([1000, 400])
            elif type(grad_index) == np.ndarray:
                # print("numpy")
                index = grad_index
                value = grad_value
                pass
            else:
                index = grad_index.__array__(
                )  # to_tensor(grad_index, place='cpu')
                value = grad_value.__array__(
                )  #to_tensor(grad_value, place='cpu')
                # index = grad_index.numpy()
                # value = grad_value.numpy()

            embeds._update(value, index)
            #time.sleep(0.001)
            if type(grad_index) == SharedArray:
                grad_index.unlink()
                grad_value.unlink()


def timer(f):
    def inner(*args, **kwargs):
        start = time.time()
        ret = f(*args, **kwargs)
        end = time.time()
        print('%s runing time: %s' % (f.__name__, end - start))

    return inner


class mp_speed_test(unittest.TestCase):
    def start_async_update(self):
        """initialize the async update
        """
        self.dtype = "float32"
        self.index = (np.random.rand(1000) * 1000).astype('int32')
        self.grad = np.random.rand(1000, 400).astype(self.dtype)
        self.index_t = paddle.to_tensor(self.index).cpu()
        self.grad_t = paddle.to_tensor(self.grad).cpu()

        self.embs = NumPyEmbedding(15000, 400, -0.05, 0.05, "./__tmp.npy",
                                   "adagrad", 0.001)

        self._async_q = mp.Queue(1)
        self._async_p = mp.Process(
            target=async_update, args=(self.embs, self._async_q))
        self._async_p.start()

    def finish_async_update(self):
        """Notify the async update process to quit
        """
        self._async_q.put((None, None))
        self._async_p.join()

    def test_pickle(self):
        self.start_async_update()

        @timer
        def test_pickle_send():
            for i in range(1000):
                self._async_q.put([self.index, self.grad])

        test_pickle_send()
        self.finish_async_update()

    def test_pure_update(self):
        self.start_async_update()

        @timer
        def test_pure_update_no_send():
            for i in range(1000):
                self.embs._update(self.grad, self.index)

        test_pure_update_no_send()
        self.finish_async_update()

    def test_deepcopy_speed(self):
        self.start_async_update()

        @timer
        def test_deepcopy_speed_no_send():
            for i in range(1000):
                grad = copy.deepcopy(self.grad)
                index = copy.deepcopy(self.index)

        test_deepcopy_speed_no_send()
        self.finish_async_update()

    def test_shareNDArray(self):
        self.start_async_update()

        @timer
        def test_share_ndarray_send():
            for i in range(1000):
                index = SharedArray.copy_from(self.index)
                grad = SharedArray.copy_from(self.grad.reshape((-1, )))
                self._async_q.put([index, grad])
            self.finish_async_update()

        test_share_ndarray_send()

    def test_paddle_shared_mem(self):
        self.start_async_update()

        index = self.index_t.cpu().clone()._share_memory()
        grad = self.grad_t.cpu().clone()._share_memory()

        @timer
        def test_paddle_shared_mem_send():
            for i in range(1000):
                # index = self.index_t.cpu().clone()._share_memory()
                index = self.index_t.cpu()._share_memory()
                # grad = self.grad_t.cpu().clone()._share_memory()
                grad = self.grad_t.cpu()._share_memory()

                self._async_q.put([index, grad])
                # paddle.fluid.core._remove_tensor_list_mmap_fds([index, grad])
            self.finish_async_update()

        test_paddle_shared_mem_send()
