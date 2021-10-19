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
"""
    StreamPool for asynchronously push or pull data in non-default CUDA streams.
    The implementation of StreamPool refers to `pygas`(https://github.com/rusty1s/pyg_autoscale).
"""

import numpy as np

import paddle
from paddle.fluid import core
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
            log.info("New Push Stream")
        return self._push_streams[idx]

    def _pull_stream(self, idx):
        if self._pull_streams[idx] is None:
            self._pull_streams[idx] = cuda.Stream()
            log.info("New Pull Stream")
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
            core.async_read(src,
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
            core.async_write(src, dst, offset, count)

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
        self._push_stream(idx).synchronize()
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
