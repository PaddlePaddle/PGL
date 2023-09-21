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
    StreamPool for asynchronously push or pull data in non-default CUDA streams.
    The implementation of StreamPool refers to `pygas`(https://github.com/rusty1s/pyg_autoscale).
"""

import numpy as np
import paddle
from paddle.framework import core
from paddle.device import cuda
from pgl.utils.logger import log


class StreamPool(paddle.nn.Layer):
    def __init__(self, pool_size, buffer_size, emb_dim):
        super().__init__()

        self.pool_size = pool_size
        self.buffer_size = buffer_size
        self.emb_dim = emb_dim

        self._push_streams = [None] * pool_size
        self._pull_streams = [None] * pool_size
        self._cpu_buffers = [None] * pool_size
        self._gpu_buffers = [None] * pool_size
        self._push_cache = [None] * pool_size
        self._pull_queue = []
        self._pull_index = -1
        self._push_index = -1

    def _push_stream(self, idx):
        if self._push_streams[idx] is None:
            self._push_streams[idx] = cuda.Stream()
        return self._push_streams[idx]

    def _pull_stream(self, idx):
        if self._pull_streams[idx] is None:
            self._pull_streams[idx] = cuda.Stream()
        return self._pull_streams[idx]

    def _cpu_buffer(self, idx):
        if self._cpu_buffers[idx] is None:
            numpy_data = np.zeros(
                (self.buffer_size, self.emb_dim), dtype=np.float32)
            self._cpu_buffers[idx] = paddle.to_tensor(
                numpy_data, place=paddle.CUDAPinnedPlace())
        return self._cpu_buffers[idx]

    def _gpu_buffer(self, idx):
        if self._gpu_buffers[idx] is None:
            self._gpu_buffers[idx] = paddle.empty(
                shape=[self.buffer_size, self.emb_dim], dtype="float32")
        return self._gpu_buffers[idx]

    @paddle.no_grad()
    def _async_pull(self, idx, src, index, offset, count):
        with cuda.stream_guard(self._pull_stream(idx)):
            async_read(src,
                       self._gpu_buffer(idx), index,
                       self._cpu_buffer(idx), offset, count)

    @paddle.no_grad()
    def async_pull(self, src, index, offset, count):
        self._pull_index = (self._pull_index + 1) % self.pool_size
        pull_data = (self._pull_index, src, index, offset, count)
        self._pull_queue.append(pull_data)
        if len(self._pull_queue) <= self.pool_size:
            self._async_pull(self._pull_index, src, index, offset, count)

    @paddle.no_grad()
    def _async_push(self, idx, src, dst, offset, count):
        with cuda.stream_guard(self._push_stream(idx)):
            async_write(src, dst, offset, count)

    @paddle.no_grad()
    def async_push(self, src, dst, offset, count):
        self._push_index = (self._push_index + 1) % self.pool_size
        self.sync_push(self._push_index)
        self._push_cache[self._push_index] = src
        self._async_push(self._push_index, src, dst, offset, count)

    @paddle.no_grad()
    def sync_pull(self):
        idx = self._pull_queue[0][0]
        cuda.synchronize()
        self._pull_stream(idx).synchronize()
        return self._gpu_buffer(idx)

    @paddle.no_grad()
    def _sync_push(self, idx):
        # TODO(daisiming): why stream.synchronize() not worked here?
        self._push_cache[idx] = None
        cuda.synchronize()

    @paddle.no_grad()
    def sync_push(self, idx=None):
        if idx is None:
            for idx in range(self.pool_size):
                self._sync_push(idx)
            self._push_index = -1
        else:
            self._sync_push(idx)

    @paddle.no_grad()
    def free_pull(self):
        self._pull_queue.pop(0)
        if len(self._pull_queue) >= self.pool_size:
            pull_data = self._pull_queue[self.pool_size - 1]
            idx, src, index, offset, count = pull_data
            self._async_pull(idx, src, index, offset, count)
        elif len(self._pull_queue) == 0:
            self._pull_index = -1

    def forward(self, *args, **kwargs):
        raise NotImplementedError


def async_read(src, dst, index, cpu_buffer, offset, count):
    """This api provides a way to read from pieces of source tensor to destination tensor 
    asynchronously. In which, we use `index`, `offset` and `count` to determine where 
    to read. `index` means the index position of src tensor we want to read. `offset` 
    and `count` means the begin points and length of pieces of src tensor we want to read. 
    To be noted, the copy process will run asynchronously from pin memory to cuda place. 
    
    We can simply remember this as "cuda async_read from pin_memory". We should run this 
    api under GPU version PaddlePaddle.

    Args:

        src (Tensor): The source tensor, and the data type should be `float32` currently. 
                      Besides, `src` should be placed on CUDAPinnedPlace.

        dst (Tensor): The destination tensor, and the data type should be `float32` currently. 
                      Besides, `dst` should be placed on CUDAPlace. The shape of `dst` should 
                      be the same with `pin_src` except for the first dimension.

        index (Tensor): The index tensor, and the data type should be `int64` currently. 
                      Besides, `index` should be on CPUPlace. The shape of `index` should 
                      be one-dimensional.

        cpu_buffer (Tensor): The cpu_buffer tensor, used to buffer index copy tensor temporarily.
                      The data type should be `float32` currently, and should be placed 
                      on CUDAPinnedPlace. The shape of `cpu_buffer` should be the same with 
                      `src` except for the first dimension.

        offset (Tensor): The offset tensor, and the data type should be `int64` currently. 
                      Besides, `offset` should be placed on CPUPlace. The shape of `offset` 
                      should be one-dimensional.

        count (Tensor): The count tensor, and the data type should be `int64` currently. 
                      Besides, `count` should be placed on CPUPlace. The shape of `count` 
                      should be one-dimensinal.

    Examples:

        import numpy as np
        import paddle
        from pgl.utils.stream_pool import async_read

        src = paddle.rand(shape=[100, 50, 50], dtype="float32").pin_memory()
        dst = paddle.empty(shape=[100, 50, 50], dtype="float32")
        offset = paddle.to_tensor(
            np.array([0, 60], dtype="int64"), place=paddle.CPUPlace())
        count = paddle.to_tensor(
            np.array([40, 60], dtype="int64"), place=paddle.CPUPlace())
        cpu_buffer = paddle.empty(shape=[50, 50, 50], dtype="float32").pin_memory()
        index = paddle.to_tensor(
            np.array([1, 3, 5, 7, 9], dtype="int64")).cpu()
        async_read(src, dst, index, cpu_buffer, offset, count)

    """

    core.async_read(src, dst, index, cpu_buffer, offset, count)


def async_write(src, dst, offset, count):
    """This api provides a way to write pieces of source tensor to destination tensor 
    asynchronously. In which, we use `offset` and `count` to determine copy to where. 
    `offset` means the begin points of the copy destination of `dst`, and `count` 
    means the lengths of the copy destination of `dst`. To be noted, the copy process 
    will run asynchronously from cuda to pin memory.
    
    We can simply remember this as "gpu async_write to pin_memory". We should run this
    api under GPU version PaddlePaddle.

    Args:
  
        src (Tensor): The source tensor, and the data type should be `float32` currently. 
                      Besides, `src` should be placed on CUDAPlace.

        dst (Tensor): The destination tensor, and the data type should be `float32` currently. 
                      Besides, `dst` should be placed on CUDAPinnedPlace. The shape of 
                      `dst` should be the same with `src` except for the first dimension. 

        offset (Tensor): The offset tensor, and the data type should be `int64` currently. 
                      Besides, `offset` should be placed on CPUPlace. The shape of `offset` 
                      should be one-dimensional. 
    
        count (Tensor): The count tensor, and the data type should be `int64` currently. 
                      Besides, `count` should be placed on CPUPlace. The shape of `count` 
                      should be one-dimensinal.

    Examples:

        import numpy as np
        import paddle
        from pgl.utils.stream_pool import async_write
   
        src = paddle.rand(shape=[100, 50, 50])
        dst = paddle.empty(shape=[200, 50, 50]).pin_memory()
        offset = paddle.to_tensor(
             np.array([0, 60], dtype="int64"), place=paddle.CPUPlace())
        count = paddle.to_tensor(
             np.array([40, 60], dtype="int64"), place=paddle.CPUPlace())
        async_write(src, dst, offset, count)

    """

    core.async_write(src, dst, offset, count)
