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
from collections import defaultdict
from _thread import start_new_thread
from multiprocessing import Queue, Process

import paddle
import numpy as np
from tqdm import tqdm


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

    for arg in vars(args):
        logging.info('{:20}:{}'.format(arg, getattr(args, arg)))


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
            print(f'[{name}] start...')
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
            print(f'[{name}] finished! It takes {cost_str} s')
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


def calculate_metrics(scores, corr_idxs, filter_list):
    """Calculate metrics according to scores
    """
    logs = []
    for i in range(scores.shape[0]):
        rank = (scores[i] > scores[i][corr_idxs[i]]).astype('float32')
        if filter_list is not None:
            mask = paddle.ones(rank.shape, dtype='float32')
            mask[filter_list[i]] = 0.
            rank = rank * mask
        rank = paddle.sum(rank) + 1
        logs.append({
            'MRR': 1.0 / rank,
            'MR': float(rank),
            'HITS@1': 1.0 if rank <= 1 else 0.0,
            'HITS@3': 1.0 if rank <= 3 else 0.0,
            'HITS@10': 1.0 if rank <= 10 else 0.0,
        })
    return logs


@timer_wrapper('evaluation')
def evaluate(model,
             loader,
             evaluate_mode='test',
             filter_dict=None,
             save_path='./tmp/'):
    """evaluate
    """
    model.eval()
    with paddle.no_grad():
        h_output = defaultdict(list)
        t_output = defaultdict(list)
        h_metrics = []
        t_metrics = []
        dataset_mode = 'normal'
        output = {'h,r->t': {}, 't,r->h': {}, 'average': {}}

        for mode, (h, r, t, cand), corr_idx in tqdm(loader):
            dataset_mode = mode
            if mode == 'wiki':
                score = model.predict(h, r, cand)
                rank = paddle.argsort(score, axis=1, descending=True)
                t_output['t_pred_top10'].append(rank[:, :10].cpu())
                t_output['t_correct_index'].append(corr_idx.cpu())
            else:
                t_score = model.predict(h, r, cand, mode='tail')
                h_score = model.predict(t, r, cand, mode='head')

                if filter_dict is not None:
                    h_filter_list = [
                        filter_dict['head'][(ti, ri)]
                        for ti, ri in zip(t.numpy(), r.numpy())
                    ]
                    t_filter_list = [
                        filter_dict['tail'][(hi, ri)]
                        for hi, ri in zip(h.numpy(), r.numpy())
                    ]
                else:
                    h_filter_list = None
                    t_filter_list = None

                h_metrics += calculate_metrics(h_score, h, h_filter_list)
                t_metrics += calculate_metrics(t_score, t, t_filter_list)

        if dataset_mode == 'wiki':
            for key, value in h_output.items():
                output['t,r->h'][key] = np.concatenate(value, axis=0)
            for key, value in t_output.items():
                output['h,r->t'][key] = np.concatenate(value, axis=0)
            paddle.save(output, save_path)
        else:
            for metric in h_metrics[0].keys():
                output['t,r->h'][metric] = np.mean(
                    [x[metric] for x in h_metrics])
                output['h,r->t'][metric] = np.mean(
                    [x[metric] for x in t_metrics])
                output['average'][metric] = (
                    output['t,r->h'][metric] + output['h,r->t'][metric]) / 2
            print('-------------- %s result --------------' % evaluate_mode)
            print('t,r->h  |', ' '.join(
                ['{}: {}'.format(k, v) for k, v in output['t,r->h'].items()]))
            print('h,r->t  |', ' '.join(
                ['{}: {}'.format(k, v) for k, v in output['h,r->t'].items()]))
            print('average |', ' '.join(
                ['{}: {}'.format(k, v) for k, v in output['average'].items()]))
            print('-----------------------------------------')


def gram_schimidt_process(embeds, num_elem, use_scale):
    """ Orthogonalize embeddings
    """
    num_embed = embeds.shape[0]
    assert embeds.shape[1] == num_elem
    assert embeds.shape[2] == (num_elem + int(use_scale))
    if use_scale:
        scales = embeds[:, :, -1]
        embeds = embeds[:, :, :num_elem]

    u = [embeds[:, 0]]
    uu = [0] * num_elem
    uu[0] = (u[0] * u[0]).sum(axis=-1)
    u_d = embeds[:, 1:]
    ushape = (num_embed, 1, -1)
    for i in range(1, num_elem):
        tmp_a = (embeds[:, i:] * u[i - 1].reshape(ushape)).sum(axis=-1)
        tmp_b = uu[i - 1].reshape((num_embed, -1))
        tmp_u = (tmp_a / tmp_b).reshape((num_embed, -1, 1))
        u_d = u_d - u[-1].reshape(ushape) * tmp_u
        u_i = u_d[:, 0]
        if u_d.shape[1] > 1:
            u_d = u_d[:, 1:]
        uu[i] = (u_i * u_i).sum(axis=-1)
        u.append(u_i)

    u = np.stack(u, axis=1)
    u_norm = np.linalg.norm(u, axis=-1, keepdims=True)
    u = u / u_norm
    if use_scale:
        u = np.concatenate([u, scales.reshape((num_embed, -1, 1))], axis=-1)
    return u
