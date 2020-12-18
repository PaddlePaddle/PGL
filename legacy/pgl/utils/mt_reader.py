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
"""Optimized Multithreading Reader for PaddlePaddle
"""
import logging
log = logging.getLogger(__name__)
import threading
import queue
import copy
import numpy as np
import time
import paddle.fluid as fluid


def multithreading_reader(readers, queue_size=1000):
    """
    multithreading_reader use python multi thread to read data from readers
    and then use queue to merge all
    data. The process number is equal to the number of input readers, each
    process call one reader.
    CPU usage rate won't go over 100% with GIL. 
    you need to create multiple readers first, these readers should be independent
    to each other so that each process can work independently.
    An example:
    .. code-block:: python
        reader0 = reader(["file01", "file02"])
        reader1 = reader(["file11", "file12"])
        reader1 = reader(["file21", "file22"])
        reader = multithreading_reader([reader0, reader1, reader2],
            queue_size=100)
    """

    assert type(readers) is list and len(readers) > 0

    def _read_into_queue(reader, queue):
        """read_into_queue"""
        for sample in reader():
            if sample is None:
                raise ValueError("sample has None")
            queue.put(sample)
        queue.put(None)

    def queue_reader():
        """queue_reader"""
        output_queue = queue.Queue(queue_size)
        thread_pool = []
        thread_num = 0
        for reader in readers:
            p = threading.Thread(
                target=_read_into_queue, args=(reader, output_queue))
            p.daemon = True
            p.start()
            thread_pool.append(p)
            thread_num += 1

        while True:
            ret = output_queue.get()
            if ret is not None:
                yield ret
            else:
                thread_num -= 1
                if thread_num == 0:
                    break

        for thread in thread_pool:
            thread.join()

    return queue_reader
