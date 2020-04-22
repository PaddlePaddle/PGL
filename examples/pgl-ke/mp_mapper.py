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
This file aims to use multiprocessing to do following process.
    `
    for data in reader():
        yield func(data)
    `
"""
#encoding=utf8
import numpy as np
import multiprocessing as mp
import traceback
from pgl.utils.logger import log


def mp_reader_mapper(reader, func, num_works=4):
    """
    This function aims to use multiprocessing to do following process.
    `
    for data in reader():
        yield func(data)
    `
    The data in_stream is the `reader`, the mapper is map the in_stream to
    an out_stream.
    Please ensure the `func` have specific return value, not `None`!
    :param reader: the data iterator
    :param func: the map func
    :param num_works: number of works
    :return: an new iterator
    """

    def _read_into_pipe(func, conn):
        """
        read into pipe, and use the `func` to get final data.
        """
        while True:
            data = conn.recv()
            if data is None:
                conn.send(None)
                conn.close()
                break
            conn.send(func(data))

    def pipe_reader():
        """pipe_reader"""
        conns = []
        all_process = []
        for w in range(num_works):
            parent_conn, child_conn = mp.Pipe()
            conns.append(parent_conn)
            p = mp.Process(target=_read_into_pipe, args=(func, child_conn))
            p.start()
            all_process.append(p)

        data_iter = reader()
        if not hasattr(data_iter, "__next__"):
            __next__ = data_iter.next
        else:
            __next__ = data_iter.__next__

        def next_data():
            """next_data"""
            _next = None
            try:
                _next = __next__()
            except StopIteration:
                # log.debug(traceback.format_exc())
                pass
            except Exception:
                log.debug(traceback.format_exc())
            return _next

        for i in range(num_works):
            conns[i].send(next_data())

        finish_num = 0
        finish_flag = np.zeros(len(conns), dtype="int32")
        while finish_num < num_works:
            for conn_id, conn in enumerate(conns):
                if finish_flag[conn_id] > 0:
                    continue
                sample = conn.recv()
                if sample is None:
                    finish_num += 1
                    conn.close()
                    finish_flag[conn_id] = 1
                else:
                    yield sample
                    conns[conn_id].send(next_data())

    return pipe_reader
