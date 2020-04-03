# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""Evaluation"""
import os
import sys
import time
import argparse
import numpy as np


def z_score(x, mean, std):
    """z_score"""
    return (x - mean) / std


def z_inverse(x, mean, std):
    """The inverse of function z_score"""
    return x * std + mean


def MAPE(v, v_):
    """Mean absolute percentage error."""
    return np.mean(np.abs(v_ - v) / (v + 1e-5))


def RMSE(v, v_):
    """Mean squared error."""
    return np.sqrt(np.mean((v_ - v)**2))


def MAE(v, v_):
    """Mean absolute error."""
    return np.mean(np.abs(v_ - v))


def evaluation(y, y_, x_stats):
    """Calculate MAPE, MAE and RMSE between ground truth and prediction."""
    dim = len(y_.shape)

    if dim == 3:
        # single_step case
        v = z_inverse(y, x_stats['mean'], x_stats['std'])
        v_ = z_inverse(y_, x_stats['mean'], x_stats['std'])
        return np.array([MAPE(v, v_), MAE(v, v_), RMSE(v, v_)])
    else:
        # multi_step case
        tmp_list = []
        # y -> [time_step, batch_size, n_route, 1]
        y = np.swapaxes(y, 0, 1)
        # recursively call
        for i in range(y_.shape[0]):
            tmp_res = evaluation(y[i], y_[i], x_stats)
            tmp_list.append(tmp_res)
        return np.concatenate(tmp_list, axis=-1)
