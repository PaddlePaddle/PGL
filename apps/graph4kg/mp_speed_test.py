import unittest
import time
import multiprocessing as mp
import numpy as np
import paddle
from utils.helper import thread_wrapper 
from models.shared_numpy import SharedArray

paddle.ones([2,2])
print("\n"*10)

@thread_wrapper
def async_update(embeds, queue):
    while True:
        (a,b) = queue.get()
        if a is None:
            return

def timer(f):
    def inner(*args, **kwargs):
        start = time.time()
        ret = f(*args, **kwargs) 
        end = time.time()
        print('%s runing time: %s' % (f.__name__, end - start))

    return inner


class mp_speed_test(unittest.TestCase):
    def start_async_update(self):
        """initialize the async update
        """
        self.dtype = "float32"
        self.index = np.random.rand(1024).astype(self.dtype)
        self.grad = np.random.rand(1024,256).astype(self.dtype)
        self.index_t = paddle.to_tensor(self.index).cpu()
        self.grad_t = paddle.to_tensor(self.grad).cpu()

        self._async_q = mp.Queue(1)
        self._async_p = mp.Process(
            target=async_update, args=(self, self._async_q))
        self._async_p.start()

    def finish_async_update(self):
        """Notify the async update process to quit
        """
        self._async_q.put((None, None))
        self._async_p.join()
    
    def test_pickle(self):
        self.start_async_update()

        @timer
        def test_pickle_send():
            for i in range(1000):
                self._async_q.put([self.index, self.grad])
        test_pickle_send()

        self.finish_async_update()

    def test_shareNDArray(self):
        self.start_async_update()
        
        @timer
        def test_share_ndarray_send():
            for i in range(1000):
                index = SharedArray.copy_from(self.index)
                grad = SharedArray.copy_from(self.grad.reshape((-1,)))
                self._async_q.put([index, grad])
        test_share_ndarray_send()
        self.finish_async_update()

    def test_paddle_shared_mem(self):
        self.start_async_update()

        @timer 
        def test_paddle_shared_mem_send():
            for i in range(1000):
                index = self.index_t.cpu()._share_memory()
                grad = self.grad_t.cpu()._share_memory()
                self._async_q.put([index, grad])
        test_paddle_shared_mem_send()
        self.finish_async_update()

