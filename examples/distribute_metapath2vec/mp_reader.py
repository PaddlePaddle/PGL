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
import multiprocessing
import numpy as np
import time

import paddle.fluid as fluid
import pyarrow


def _serialize_serializable(obj):
    """Serialize Feed Dict
    """
    return {"type": type(obj), "data": obj.__dict__}


def _deserialize_serializable(obj):
    """Deserialize Feed Dict
    """

    val = obj["type"].__new__(obj["type"])
    val.__dict__.update(obj["data"])
    return val


context = pyarrow.default_serialization_context()

context.register_type(
    object,
    "object",
    custom_serializer=_serialize_serializable,
    custom_deserializer=_deserialize_serializable)


def serialize_data(data):
    """serialize_data"""
    return pyarrow.serialize(data, context=context).to_buffer().to_pybytes()


def deserialize_data(data):
    """deserialize_data"""
    return pyarrow.deserialize(data, context=context)


def multiprocess_reader(readers, use_pipe=True, queue_size=1000):
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

    def _read_into_pipe(reader, conn):
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
                target=_read_into_pipe, args=(reader, child_conn))
            p.start()

        reader_num = len(readers)
        finish_num = 0
        conn_to_remove = []
        finish_flag = np.zeros(len(conns), dtype="int32")
        while finish_num < reader_num:
            for conn_id, conn in enumerate(conns):
                if finish_flag[conn_id] > 0:
                    continue
                buff = conn.recv()
                now = time.time()
                sample = deserialize_data(buff)
                out = time.time() - now
                if sample is None:
                    finish_num += 1
                    conn.close()
                    finish_flag[conn_id] = 1
                else:
                    yield sample

    if use_pipe:
        return pipe_reader
    else:
        return queue_reader
