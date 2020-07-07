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
"""
The source code of anonymousmemmap is
from: https://github.com/rainwoodman/sharedmem

Many tanks!
"""
import numpy
import mmap

try:
    # numpy >= 1.16
    _unpickle_ctypes_type = numpy.ctypeslib.as_ctypes_type(numpy.dtype('|u1'))
except:
    # older version numpy < 1.16
    _unpickle_ctypes_type = numpy.ctypeslib._typecodes['|u1']


def __unpickle__(ai, dtype):
    dtype = numpy.dtype(dtype)
    tp = _unpickle_ctypes_type * 1

    # if there are strides, use strides, otherwise the stride is the itemsize of dtype
    if ai['strides']:
        tp *= ai['strides'][-1]
    else:
        tp *= dtype.itemsize

    for i in numpy.asarray(ai['shape'])[::-1]:
        tp *= i

    # grab a flat char array at the sharemem address, with length at least contain ai required
    ra = tp.from_address(ai['data'][0])
    buffer = numpy.ctypeslib.as_array(ra).ravel()
    # view it as what it should look like
    shm = numpy.ndarray(
        buffer=buffer, dtype=dtype, strides=ai['strides'],
        shape=ai['shape']).view(type=anonymousmemmap)
    return shm


class anonymousmemmap(numpy.memmap):
    """ Arrays allocated on shared memory.
        The array is stored in an anonymous memory map that is shared between child-processes.
    """

    def __new__(subtype, shape, dtype=numpy.uint8, order='C'):

        descr = numpy.dtype(dtype)
        _dbytes = descr.itemsize

        shape = numpy.atleast_1d(shape)
        size = 1
        for k in shape:
            size *= k

        bytes = int(size * _dbytes)

        if bytes > 0:
            mm = mmap.mmap(-1, bytes)
        else:
            mm = numpy.empty(0, dtype=descr)
        self = numpy.ndarray.__new__(
            subtype, shape, dtype=descr, buffer=mm, order=order)
        self._mmap = mm
        return self

    def __array_wrap__(self, outarr, context=None):
        # after ufunc this won't be on shm!
        return numpy.ndarray.__array_wrap__(
            self.view(numpy.ndarray), outarr, context)

    def __reduce__(self):
        return __unpickle__, (self.__array_interface__, self.dtype)


def copy_to_shm(a):
    """ Copy an array to the shared memory.
        Notes
        -----
        copy is not always necessary because the private memory is always copy-on-write.
        Use :code:`a = copy(a)` to immediately dereference the old 'a' on private memory
    """
    shared = anonymousmemmap(a.shape, dtype=a.dtype)
    shared[:] = a[:]
    return shared


def ToShareMemGraph(graph):
    """Copy the graph object to anonymous shared memory.
    """

    def share_feat(feat):
        for key in feat:
            feat[key] = copy_to_shm(feat[key])

    def share_adj_index(index):
        if index is not None:
            index._degree = copy_to_shm(index._degree)
            index._sorted_u = copy_to_shm(index._sorted_u)
            index._sorted_v = copy_to_shm(index._sorted_v)
            index._sorted_eid = copy_to_shm(index._sorted_eid)
            index._indptr = copy_to_shm(index._indptr)

    graph._edges = copy_to_shm(graph._edges)
    share_adj_index(graph._adj_src_index)
    share_adj_index(graph._adj_dst_index)
    share_feat(graph._node_feat)
    share_feat(graph._edge_feat)
