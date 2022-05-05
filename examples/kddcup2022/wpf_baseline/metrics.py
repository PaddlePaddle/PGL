# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np


def ignore_zeros(predictions, grounds):
    """
    Desc:
        Ignore the zero values for evaluation
    Args:
        predictions:
        grounds:
    Returns:
        Predictions and ground truths
    """
    predictions[np.where(grounds == 0)] = 0
    grounds[np.where(grounds == 0)] = 0
    return predictions, grounds


def rse(pred, ground_truth):
    """
    Desc:
        Root square error
    Args:
        pred:
        ground_truth: ground truth vector
    Returns:
        RSE value
    """
    return np.sqrt(np.sum((ground_truth - pred)**2)) / np.sqrt(
        np.sum((ground_truth - ground_truth.mean())**2))


def corr(pred, gt):
    """
    Desc:
        Correlation between the prediction and ground truth
    Args:
        pred:
        gt: ground truth vector
    Returns:
        Correlation
    """
    u = ((gt - gt.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((gt - gt.mean(0))**2 * (pred - pred.mean(0))**2).sum(0))
    return (u / d).mean(-1)


def mae(pred, gt):
    """
    Desc:
        Mean Absolute Error
    Args:
        pred:
        gt: ground truth vector
    Returns:
        MAE value
    """
    return np.mean(np.abs(pred - gt))


def mse(pred, gt):
    """
    Desc:
        Mean Square Error
    Args:
        pred:
        gt: ground truth vector
    Returns:
        MSE value
    """
    return np.mean((pred - gt)**2)


def rmse(pred, gt):
    """
    Desc:
        Root Mean Square Error
    Args:
        pred:
        gt: ground truth vector
    Returns:
        RMSE value
    """
    return np.sqrt(mse(pred, gt))


def mape(pred, gt):
    """
    Desc:
        Mean Absolute Percentage Error
    Args:
        pred:
        gt: ground truth vector
    Returns:
        MAPE value
    """
    return np.mean(np.abs((pred - gt) / gt))


def mspe(pred, gt):
    """
    Desc:
        Mean Square Percentage Error
    Args:
        pred:
        gt: ground truth vector
    Returns:
        MSPE value
    """
    return np.mean(np.square((pred - gt) / gt))


def regressor_metrics(pred, gt):
    """
    Desc:
        Some common metrics for regression problems
    Args:
        pred:
        gt: ground truth vector
    Returns:
        A tuple of metrics
    """
    pred, gt = ignore_zeros(pred, gt)
    _mae = mae(pred, gt)
    _mse = mse(pred, gt)
    _rmse = rmse(pred, gt)
    return _mae, _mse, _rmse
