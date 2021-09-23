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

import mmap

import posix_ipc
import numpy as np


class SharedArray(object):
    """Wrapper for sharing numpy.ndarray using POSIX shared memory.
    """

    def __init__(self, shape, dtype=np.float64, name=None):
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        if name:
            self._shm = posix_ipc.SharedMemory(name)
        else:
            self._shm = posix_ipc.SharedMemory(
                None, posix_ipc.O_CREX, size=size)
        self._buf = mmap.mmap(self._shm.fd, size)
        self.array = np.ndarray(shape, dtype, self._buf, order='C')

    @classmethod
    def copy_from(cls, array):
        """Create a SharedArray object from array
        """
        shared_array = cls(array.shape, array.dtype)
        shared_array.array[:] = array
        return shared_array

    def unlink(self):
        """Mark the shared memory for deletion
        """
        self._shm.unlink()

    def __del__(self):
        self._buf.close()
        self._shm.close_fd()

    def __getstate__(self):
        return self.array.shape, self.array.dtype, self._shm.name

    def __setstate__(self, state):
        self.__init__(*state)
