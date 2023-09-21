# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""Optimized Multiprocessing Reader for PaddlePaddle
"""

import logging
log = logging.getLogger(__name__)
import multiprocessing
import copy
try:
    import ujson as json
except:
    log.info("ujson not install, fail back to use json instead")
    import json
import numpy as np
import time
from multiprocessing import Queue
import threading
from collections import namedtuple

_np_serialized_data = namedtuple("_np_serialized_data",
                                 ["value", "shape", "dtype"])


def serialize_data(data):
    """serialize_data"""
    if data is None:
        return None
    return numpy_serialize_data(data)  #, ensure_ascii=False)


def index_iter(data):
    """return indexing iter"""
    if isinstance(data, list):
        return range(len(data))
    elif isinstance(data, dict):
        return data.keys()


def numpy_serialize_data(data):
    """serialize_data"""
    ret_data = copy.deepcopy(data)

    if isinstance(ret_data, (dict, list)):
        for key in index_iter(ret_data):
            if isinstance(ret_data[key], np.ndarray):
                ret_data[key] = _np_serialized_data(
                    value=ret_data[key].tobytes(),
                    shape=list(ret_data[key].shape),
                    dtype="%s" % ret_data[key].dtype)
    return ret_data


def numpy_deserialize_data(data):
    """deserialize_data"""
    if data is None:
        return None

    if isinstance(data, (dict, list)):
        for key in index_iter(data):
            if isinstance(data[key], _np_serialized_data):
                data[key] = np.frombuffer(
                    buffer=data[key].value,
                    dtype=data[key].dtype).reshape(data[key].shape)
    return data


def deserialize_data(data):
    """deserialize_data"""
    return numpy_deserialize_data(data)


def multiprocess_reader(readers, use_pipe=True, queue_size=1000, pipe_size=10):
    """
    multiprocess_reader use python multi process to read data from readers
    and then use multiprocess.Queue or multiprocess.Pipe to merge all
    data. The process number is equal to the number of input readers, each
    process call one reader.
    Multiprocess.Queue require the rw access right to /dev/shm, some
    platform does not support.
    you need to create multiple readers first, these readers should be independent
    to each other so that each process can work independently.
    An example:
    .. code-block:: python
        reader0 = reader(["file01", "file02"])
        reader1 = reader(["file11", "file12"])
        reader1 = reader(["file21", "file22"])
        reader = multiprocess_reader([reader0, reader1, reader2],
            queue_size=100, use_pipe=False)
    """

    assert type(readers) is list and len(readers) > 0

    def _read_into_queue(reader, queue):
        """read_into_queue"""
        for sample in reader():
            if sample is None:
                raise ValueError("sample has None")
            queue.put(serialize_data(sample))
        queue.put(serialize_data(None))

    def queue_reader():
        """queue_reader"""
        queues = []
        for reader in readers:
            queue = multiprocessing.Queue(queue_size)
            queues.append(queue)
            p = multiprocessing.Process(
                target=_read_into_queue, args=(reader, queue))
            try:
                p.start()
            except:
                raise RuntimeError(
                    f"The program met some problems. If your system is Mac OS and python >= 3.8, "
                    f"please checkout https://github.com/PaddlePaddle/PGL/issues/305 to fix the problem."
                )

        reader_num = len(readers)
        alive_queue_indices = [i for i in range(reader_num)]
        while len(alive_queue_indices) > 0:
            for alive_queue_index in [i for i in alive_queue_indices]:
                sample = deserialize_data(queues[alive_queue_index].get())
                if sample is None:
                    alive_queue_indices.remove(alive_queue_index)
                else:
                    yield sample

    def _read_into_pipe(reader, conn, max_pipe_size):
        """read_into_pipe"""
        for sample in reader():
            if sample is None:
                raise ValueError("sample has None!")
            conn.send(serialize_data(sample))
        conn.send(serialize_data(None))
        conn.close()

    def pipe_reader():
        """pipe_reader"""
        conns = []
        for reader in readers:
            parent_conn, child_conn = multiprocessing.Pipe()
            conns.append(parent_conn)
            p = multiprocessing.Process(
                target=_read_into_pipe, args=(reader, child_conn, pipe_size))
            try:
                p.start()
            except:
                raise RuntimeError(
                    f"The program met some problems. If your system is Mac OS and python >= 3.8, "
                    f"please checkout https://github.com/PaddlePaddle/PGL/issues/305 to fix the problem."
                )

        reader_num = len(readers)
        conn_to_remove = []
        finish_flag = np.zeros(len(conns), dtype="int32")

        alive_conn_indices = [i for i in range(reader_num)]
        while len(alive_conn_indices) > 0:
            for alive_conn_index in [i for i in alive_conn_indices]:
                sample = deserialize_data(conns[alive_conn_index].recv())
                if sample is None:
                    conns[alive_conn_index].close()
                    alive_conn_indices.remove(alive_conn_index)
                else:
                    yield sample

    if use_pipe:
        return pipe_reader
    else:
        return queue_reader
