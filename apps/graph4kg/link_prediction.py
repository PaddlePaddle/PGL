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
import sys
import paddle
import numpy as np
from tqdm import tqdm


def get_metric(ranks, corr_idx):
    """Get metric values
    """
    ranks = paddle.to_tensor(ranks)
    corr_idx = paddle.to_tensor(corr_idx)
    max_index = ranks.shape[1]
    corr_ranks = paddle.ones(corr_idx.shape) * max_index
    x = paddle.nonzero(ranks == corr_idx.unsqueeze(-1))
    corr_ranks[x[:, 0]] = x[:, 1] + 1
    metric = {
        'MRR': paddle.mean(1.0 / corr_ranks),
        'MR': paddle.mean(corr_ranks),
        'HITS@1': paddle.mean((corr_ranks <= 1).astype('float')),
        'HITS@3': paddle.mean((corr_ranks <= 3).astype('float')),
        'HITS@10': paddle.mean((corr_ranks <= 10).astype('float'))
    }
    for k in metric:
        metric[k] = metric[k].cpu().numpy()
    return metric


def link_prediction(path, label, mode='normal', evaluator=None):
    """Link Prediction
    """
    rst_dict = paddle.load(path)
    print(('=' * 20) + label + ('=' * 20))

    if mode == 'wiki':
        if evaluator is None:
            from ogb.lsc import WikiKG90MEvaluator
            evaluator = WikiKG90MEvaluator()
        metrics = evaluator.eval(rst_dict)
        return metrics
    else:
        h_metric = None
        t_metric = None

        if 'h,r->t' in rst_dict:
            h_metric = get_metric(rst_dict['h,r->t']['rank'],
                                  rst_dict['h,r->t']['corr'])
            info = '|'.join(
                [' %s: %.3f ' % (k, v) for k, v in h_metric.items()])
            print('head' + info)

        if 't,r->h' in rst_dict:
            t_metric = get_metric(rst_dict['t,r->h']['rank'],
                                  rst_dict['t,r->h']['corr'])
            info = '|'.join(
                [' %s: %.3f ' % (k, v) for k, v in t_metric.items()])
            print('tail' + info)

        if h_metric is not None and t_metric is not None:
            metric = dict(
                [(k, (h_metric[k] + v)) for k, v in t_metric.items()])
            info = '|'.join(
                [' %s: %.3f ' % (k, v / 2.) for k, v in metric.items()])
            print('both' + info)


if __name__ == '__main__':
    path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else 'normal'
    if mode == 'wiki':
        from ogb.lsc import WikiKG90MEvaluator
        evaluator = WikiKG90MEvaluator()
    else:
        evaluator = None

    files = [x for x in os.listdir(path) if x.endswith('.pkl')]
    steps = [
        int(x.strip('.pkl').split('_')[-1]) for x in files if 'valid' in x
    ]
    steps.sort()
    for step in steps:
        step_path = os.path.join(path, 'valid_%d.pkl' % step)
        link_prediction(step_path, 'valid_%d' % step, evaluator)

    if 'test.pkl' in files:
        link_prediction(os.path.join(path, 'test.pkl'), 'test', evaluator)
