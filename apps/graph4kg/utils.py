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

import os
import csv
import math
import json
import time
import random
import logging
import functools
import traceback
from multiprocessing import Queue, Process
from threading import Thread
from _thread import start_new_thread

import paddle
import numpy as np

# from pgl.utils.mp_reader import deserialize_data


def set_seed(seed):
    """Set seed for reproduction

    Execute :code:`export FLAGS_cudnn_deterministic=True` before training command.

    """
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def set_logger(args):
    """Write logs to console and log file
    """
    log_file = os.path.join(args.save_path, 'train.log')
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='a+')
    if args.print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)


def print_log(step, interval, log, timer, t_step):
    """Print log
    """
    time_sum = time.time() - t_step
    logging.info('step: %d, loss: %.5f, reg: %.4e, speed: %.2f steps/s' %
                 (step, log['loss'] / interval, log['reg'] / interval,
                  interval / time_sum))
    logging.info('timer | sample: %f, forward: %f, backward: %f, update: %f' %
                 (timer['sample'], timer['forward'], timer['backward'],
                  timer['update']))


def adjust_args(args):
    """Adjust arguments for compatiblity
    """
    args.save_path = prepare_save_path(args)
    # set the path to shm, the performance will be better.
    args.embs_path = os.path.join(args.save_path, '__cpu_embedding.npy')
    # make the batch_size divisible by the neg_sample_size for chunk-based negative sampling
    batch_size = args.batch_size
    neg_sample_size = args.neg_sample_size
    if neg_sample_size < batch_size and batch_size % neg_sample_size != 0:
        batch_size = int(
            math.ceil(batch_size / neg_sample_size) * neg_sample_size)
        print('batch size {} is not divisible by negative sample size {}'.
              format(args.batch_size, args.neg_sample_size))
        print('Change the batch size to {}'.format(batch_size))
        args.batch_size = batch_size
    return args


def uniform(low, high, size, dtype=np.float32):
    """Memory efficient uniform implementation
    """
    rng = np.random.default_rng(0)
    out = (high - low) * rng.random(size, dtype=dtype) + low
    return out


def timer_wrapper(name):
    """Time counter wrapper
    """

    def decorate(func):
        """decorate func
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """wrapper func
            """
            print('[{}] start...'.format(name))
            ts = time.time()
            result = func(*args, **kwargs)
            te = time.time()
            costs = te - ts
            if costs < 1e-4:
                cost_str = '%f sec' % costs
            elif costs > 3600:
                cost_str = '%.4f sec (%.4f hours)' % (costs, costs / 3600.)
            else:
                cost_str = '%.4f sec' % costs
            print('[%s] finished! It takes %s' % (name, cost_str))
            return result

        return wrapper

    return decorate


def thread_wrapper(func):
    """Wrapped func for multiprocessing.Process
    """

    @functools.wraps(func)
    def decorate(*args, **kwargs):
        """decorate func
        """
        queue = Queue()

        def _queue_func():
            exception, trace, result = None, None, None
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                exception = e
                trace = traceback.format_exc()
            queue.put((result, exception, trace))

        start_new_thread(_queue_func, ())
        result, exception, trace = queue.get()
        if exception is None:
            return result
        else:
            assert isinstance(exception, Exception)
            raise exception.__class__(trace)

    return decorate


def thread_wrapper_(func):
    """Wrapped func for multiprocessing.Process
    """

    @functools.wraps(func)
    def decorate(*args, **kwargs):
        """decorate func
        """
        queue = Queue()

        def _queue_func():
            exception, trace, result = None, None, None
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                exception = e
                trace = traceback.format_exc()
            queue.put((result, exception, trace))

        proc = Thread(target=_queue_func)

        proc.start()
        proc.join()
        result, exception, trace = queue.get()

        if exception is None:
            return result
        else:
            assert isinstance(exception, Exception)
            raise exception.__class__(trace)

    return decorate


def to_tensor(data, place):
    return paddle.Tensor(
        value=data,
        place=paddle.fluid.core.CPUPlace(),
        persistable=False,
        zero_copy=True,
        stop_gradient=True)


@thread_wrapper
def async_update(embeds, queue):
    """Update embeddings asynchronously
    """
    while True:
        # (grad_index, grad_value) = deserialize_data(queue.get())
        # (grad_index, grad_value, grad_shape) = queue.get()
        (grad_index, grad_value, grad_2) = queue.get()
        # grad_index = to_tensor(grad_index, place='cpu')
        # grad_value = to_tensor(grad_value, place='cpu')
        if grad_index is None:
            return
        with paddle.no_grad():
            # embeds._update(grad_value.array.reshape(grad_shape), grad_index.array)
            # embeds._update(grad_value.numpy(), grad_index.numpy())
            embeds._update(grad_value, grad_index, grad_2)


def prepare_save_path(args):
    """Get model specific save path and makedirs if not exists
    """
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    folder = '{}_{}_d_{}_g_{}'.format(args.model_name, args.data_name,
                                      args.embed_dim, args.gamma)
    n = len([x for x in os.listdir(args.save_path) if x.startswith(folder)])
    folder += str(n)
    args.save_path = os.path.join(args.save_path, folder)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    else:
        raise IOError('model path %s already exists' % args.save_path)
    return args.save_path


def get_compatible_batch_size(batch_size, neg_sample_size):
    """For chunk-based batch negative sampling (optitional)
    """
    if neg_sample_size < batch_size and batch_size % neg_sample_size != 0:
        old_batch_size = batch_size
        batch_size = int(
            math.ceil(batch_size / neg_sample_size) * neg_sample_size)
        print(
            'batch size ({}) is incompatible to the negative sample size ({}). Change the batch size to {}'.
            format(old_batch_size, neg_sample_size, batch_size))
    return batch_size


def load_model_config(config_f):
    """Load configuration from config.yaml
    """
    with open(config_f, "r") as f:
        config = json.loads(f.read())

    print(config)
    return config
