#-*- coding: utf-8 -*-
import os
import sys
import time
import warnings
import queue as Queue
import numpy as np
import threading
import collections


def stream_shuffle_generator(generator, batch_size, shuffle_size=20000):
    """
    Args:
        generator: iterable dataset

        batch_size: int

        shuffle_size: int

    """
    buffer_list = []
    batch_data = []
    for examples in generator():
        if not isinstance(examples, list):
            examples = [examples]

        for data in examples:
            if len(buffer_list) < shuffle_size:
                buffer_list.append(data)
            else:
                idx = np.random.randint(0, len(buffer_list))
                batch_data.append(buffer_list[idx])
                buffer_list[idx] = data

                if len(batch_data) >= batch_size:
                    yield batch_data
                    batch_data = []

    if len(batch_data) > 0:
        yield batch_data

    if len(buffer_list) > 0:
        np.random.shuffle(buffer_list)
        start = 0
        while True:
            batch_data = buffer_list[start:(start + batch_size)]
            start += batch_size
            if len(batch_data) > 0:
                yield batch_data
            else:
                break


class AsynchronousGenerator:
    def __init__(self, generator, start=True, maxsize=0):
        self.generator = generator
        self.thread = threading.Thread(target=self._generatorcall)
        self.q = Queue.Queue(maxsize=maxsize)
        self.next = self.__next__
        if start:
            self.thread.start()

    def __next__(self):
        done, item = self.q.get()
        if done:
            raise StopIteration
        else:
            return item

    def __iter__(self):
        return self

    def __call__(self):
        return self.__iter__()

    def _generatorcall(self):
        for output in self.generator():
            self.q.put((False, output))
        self.q.put((True, None))
