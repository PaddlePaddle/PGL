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
import pickle
import json
import numpy as np
import sys
from ogb.lsc import WikiKG90MDataset, WikiKG90MEvaluator

import pdb
from collections import defaultdict
import paddle


def get_valid_result(path):
    all_file_names = os.listdir(path)
    valid_file_names = [
        name for name in all_file_names if '.pkl' in name and 'valid' in name
    ]
    steps = [
        int(name.split('.')[0].split('_')[-1]) for name in valid_file_names
        if 'valid' in name
    ]
    steps.sort()
    evaluator = WikiKG90MEvaluator()

    print(valid_file_names)
    best_valid_mrr = -1
    best_valid_idx = -1

    num_proc = 1
    for i, step in enumerate(steps):
        valid_result_dict = defaultdict(lambda: defaultdict(list))
        for proc in range(num_proc):
            valid_result_dict_proc = paddle.load(
                os.path.join(path, "valid_{}.pkl".format(step)))
            for result_dict_proc, result_dict in zip([valid_result_dict_proc],
                                                     [valid_result_dict]):
                for key in result_dict_proc['h,r->t']:
                    result_dict['h,r->t'][key].append(result_dict_proc[
                        'h,r->t'][key].numpy())
        for result_dict in [valid_result_dict]:
            for key in result_dict['h,r->t']:
                result_dict['h,r->t'][key] = np.concatenate(
                    result_dict['h,r->t'][key], 0)
        metrics = evaluator.eval(valid_result_dict)
        metric = 'mrr'
        print("valid-{} at step {}: {}".format(metric, step, metrics[metric]))
        if metrics[metric] > best_valid_mrr:
            best_valid_mrr = metrics[metric]
            best_valid_idx = i


def get_test_result(path):
    evaluator = WikiKG90MEvaluator()
    test_result_dict = defaultdict(lambda: defaultdict(list))
    num_proc = 1
    for proc in range(num_proc):
        test_result_dict_proc = paddle.load(os.path.join(path, "test.pkl"))
        for result_dict_proc, result_dict in zip([test_result_dict_proc],
                                                 [test_result_dict]):
            for key in result_dict_proc['h,r->t']:
                result_dict['h,r->t'][key].append(result_dict_proc['h,r->t'][
                    key].numpy())
    for result_dict in [test_result_dict]:
        for key in result_dict['h,r->t']:
            result_dict['h,r->t'][key] = np.concatenate(
                result_dict['h,r->t'][key], 0)
    metrics = evaluator.eval(test_result_dict)
    metric = 'mrr'
    print("Test-{}: {}".format(metric, metrics[metric]))
    print("Test Metrics:")
    print(metrics)


if __name__ == '__main__':
    path = sys.argv[1]
    mode = sys.argv[2]
    if mode == 'test':
        get_test_result(path)
    else:
        get_valid_result(path)
