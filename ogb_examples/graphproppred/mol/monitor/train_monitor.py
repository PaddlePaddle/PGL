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

import tqdm
import json
import numpy as np
import os
from datetime import datetime
import logging
from collections import defaultdict
from tensorboardX import SummaryWriter

import paddle.fluid as F
from pgl.utils.logger import log


def multi_device(reader, dev_count):
    if dev_count == 1:
        for batch in reader:
            yield batch
    else:
        batches = []
        for batch in reader:
            batches.append(batch)
            if len(batches) == dev_count:
                yield batches
                batches = []


def evaluate(exe, loader, prog, model, evaluator):
    total_labels = []
    for i in range(len(loader.dataset)):
        g, l = loader.dataset[i]
        total_labels.append(l)
    total_labels = np.vstack(total_labels)

    pred_output = []
    for feed_dict in loader:
        ret = exe.run(prog, feed=feed_dict, fetch_list=model.pred)
        pred_output.append(ret[0])

    pred_output = np.vstack(pred_output)

    result = evaluator.eval({"y_true": total_labels, "y_pred": pred_output})

    return result


def _create_if_not_exist(path):
    basedir = os.path.dirname(path)
    if not os.path.exists(basedir):
        os.makedirs(basedir)


def train_and_evaluate(exe,
                       train_exe,
                       valid_exe,
                       train_ds,
                       valid_ds,
                       test_ds,
                       train_prog,
                       valid_prog,
                       args,
                       model,
                       evaluator,
                       dev_count=1):

    global_step = 0

    timestamp = datetime.now().strftime("%Hh%Mm%Ss")
    log_path = os.path.join(args.log_dir, "tensorboard_log_%s" % timestamp)
    _create_if_not_exist(log_path)

    writer = SummaryWriter(log_path)

    best_valid_score = 0.0
    for e in range(args.epoch):
        for feed_dict in multi_device(train_ds, dev_count):
            if dev_count > 1:
                ret = train_exe.run(feed=feed_dict,
                                    fetch_list=model.metrics.vars)
                ret = [[np.mean(v)] for v in ret]
            else:
                ret = train_exe.run(train_prog,
                                    feed=feed_dict,
                                    fetch_list=model.metrics.vars)

            ret = model.metrics.parse(ret)
            if global_step % args.train_log_step == 0:
                writer.add_scalar(
                    "batch_loss", ret['loss'], global_step=global_step)
                log.info("epoch: %d | step: %d | loss: %.4f " %
                         (e, global_step, ret['loss']))

            global_step += 1
            if global_step % args.eval_step == 0:
                valid_ret = evaluate(exe, valid_ds, valid_prog, model,
                                     evaluator)
                message = "valid: "
                for key, value in valid_ret.items():
                    message += "%s %.4f | " % (key, value)
                    writer.add_scalar(
                        "eval_%s" % key, value, global_step=global_step)
                log.info(message)

                # testing
                test_ret = evaluate(exe, test_ds, valid_prog, model, evaluator)
                message = "test: "
                for key, value in test_ret.items():
                    message += "%s %.4f | " % (key, value)
                    writer.add_scalar(
                        "test_%s" % key, value, global_step=global_step)
                log.info(message)

        # evaluate after one epoch
        valid_ret = evaluate(exe, valid_ds, valid_prog, model, evaluator)
        message = "epoch %s valid: " % e
        for key, value in valid_ret.items():
            message += "%s %.4f | " % (key, value)
            writer.add_scalar("eval_%s" % key, value, global_step=global_step)
        log.info(message)

        # testing
        test_ret = evaluate(exe, test_ds, valid_prog, model, evaluator)
        message = "epoch %s test: " % e
        for key, value in test_ret.items():
            message += "%s %.4f | " % (key, value)
            writer.add_scalar("test_%s" % key, value, global_step=global_step)
        log.info(message)

        message = "epoch %s best %s result | " % (e, args.eval_metrics)
        if valid_ret[args.eval_metrics] > best_valid_score:
            best_valid_score = valid_ret[args.eval_metrics]
            best_test_score = test_ret[args.eval_metrics]

        message += "valid %.4f | test %.4f" % (best_valid_score,
                                               best_test_score)
        log.info(message)

        #  if global_step % args.save_step == 0:
        #      F.io.save_persistables(exe, os.path.join(args.save_dir, "%s" % global_step), train_prog)

    writer.close()
