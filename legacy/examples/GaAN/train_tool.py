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
import time
from pgl.utils.logger import log

def train_epoch(batch_iter, exe, program, loss, score, evaluator, epoch, log_per_step=1):
    batch = 0
    total_loss = 0.0
    total_sample = 0
    result = 0
    for batch_feed_dict in batch_iter():
        batch += 1
        batch_loss, y_pred = exe.run(program, fetch_list=[loss, score], feed=batch_feed_dict)
        
        num_samples = len(batch_feed_dict["node_index"])
        total_loss += batch_loss * num_samples
        total_sample += num_samples
        input_dict = {
            "y_true": batch_feed_dict["node_label"],
            "y_pred": y_pred
        }
        result += evaluator.eval(input_dict)["rocauc"]

    return total_loss.item()/total_sample, result/batch

def valid_epoch(batch_iter, exe, program, loss, score, evaluator, epoch, log_per_step=1):
    batch = 0
    total_sample = 0
    result = 0
    total_loss = 0.0
    for batch_feed_dict in batch_iter():
        batch += 1
        batch_loss, y_pred = exe.run(program, fetch_list=[loss, score], feed=batch_feed_dict)
        input_dict = {
            "y_true": batch_feed_dict["node_label"],
            "y_pred": y_pred
        }
        result += evaluator.eval(input_dict)["rocauc"]


        num_samples = len(batch_feed_dict["node_index"])
        total_loss += batch_loss * num_samples
        total_sample += num_samples

    return total_loss.item()/total_sample, result/batch
