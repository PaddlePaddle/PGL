# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
from queue import Queue
import threading


def serialize_data(data):
    """serialize_data"""
    if data is None:
        return None
    return numpy_serialize_data(data)  #, ensure_ascii=False)


def numpy_serialize_data(data):
    """serialize_data"""
    ret_data = {}
    for key in data:
        if isinstance(data[key], np.ndarray):
            ret_data[key] = (data[key].tobytes(), list(data[key].shape),
                             "%s" % data[key].dtype)
        else:
            ret_data[key] = data[key]
    return ret_data


def numpy_deserialize_data(data):
    """deserialize_data"""
    if data is None:
        return None
    for key in data:
        if isinstance(data[key], tuple):
            value = np.frombuffer(
                data[key][0], dtype=data[key][2]).reshape(data[key][1])
            data[key] = value
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
        queue = multiprocessing.Queue(queue_size)
        for reader in readers:
            p = multiprocessing.Process(
                target=_read_into_queue, args=(reader, queue))
            p.start()

        reader_num = len(readers)
        finish_num = 0
        while finish_num < reader_num:
            sample = deserialize_data(queue.get())
            if sample is None:
                finish_num += 1
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
            p.start()

        reader_num = len(readers)
        conn_to_remove = []
        finish_flag = np.zeros(len(conns), dtype="int32")
        start = time.time()

        def queue_worker(sub_conn, que):
            while True:
                buff = sub_conn.recv()
                sample = deserialize_data(buff)
                if sample is None:
                    que.put(None)
                    sub_conn.close()
                    break
                que.put(sample)

        thread_pool = []
        output_queue = Queue(maxsize=reader_num)
        for i in range(reader_num):
            t = threading.Thread(
                target=queue_worker, args=(conns[i], output_queue))
            t.daemon = True
            t.start()
            thread_pool.append(t)

        finish_num = 0
        while finish_num < reader_num:
            sample = output_queue.get()
            if sample is None:
                finish_num += 1
            else:
                yield sample

        for thread in thread_pool:
            thread.join()

    if use_pipe:
        return pipe_reader
    else:
        return queue_reader
