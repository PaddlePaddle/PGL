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
"""train and evaluate"""
import tqdm
import json
import numpy as np
import sys
import os
import paddle.fluid as F
from pgl.utils.log_writer import LogWriter
from ogb.linkproppred import Evaluator
from ogb.linkproppred import LinkPropPredDataset


def multi_device(reader, dev_count):
    """multi device"""
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


class OgbEvaluator(object):
    def __init__(self):
        d_name = "ogbl-ppa"
        dataset = LinkPropPredDataset(name=d_name)
        splitted_edge = dataset.get_edge_split()
        graph = dataset[0]
        self.num_nodes = graph["num_nodes"]
        self.ogb_evaluator = Evaluator(name="ogbl-ppa")

    def eval(self, scores, labels, phase):
        labels = np.reshape(labels, [-1])
        ret = {}
        pos = scores[labels > 0.5].squeeze(-1)
        neg = scores[labels < 0.5].squeeze(-1)
        for K in [10, 50, 100]:
            self.ogb_evaluator.K = K
            ret['%s_hits@%s' % (phase, K)] = self.ogb_evaluator.eval({
                'y_pred_pos': pos,
                'y_pred_neg': neg,
            })[f'hits@{K}']
        return ret


def evaluate(model, valid_exe, valid_ds, valid_prog, dev_count, evaluator,
             phase):
    """evaluate """
    cc = 0
    scores = []
    labels = []

    for feed_dict in tqdm.tqdm(
            multi_device(valid_ds.generator(), dev_count), desc='evaluating'):

        if dev_count > 1:
            output = valid_exe.run(feed=feed_dict,
                                   fetch_list=[model.logits, model.labels])
        else:
            output = valid_exe.run(valid_prog,
                                   feed=feed_dict,
                                   fetch_list=[model.logits, model.labels])
        scores.append(output[0])
        labels.append(output[1])

    scores = np.vstack(scores)
    labels = np.vstack(labels)
    ret = evaluator.eval(scores, labels, phase)
    return ret


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
                       model,
                       metric,
                       epoch=20,
                       dev_count=1,
                       train_log_step=5,
                       eval_step=10000,
                       evaluator=None,
                       output_path=None):
    """train and evaluate"""

    global_step = 0

    log_path = os.path.join(output_path, "log")
    _create_if_not_exist(log_path)

    writer = LogWriter(log_path)

    best_model = 0
    for e in range(epoch):
        for feed_dict in tqdm.tqdm(
                multi_device(train_ds.generator(), dev_count),
                desc='Epoch %s' % e):
            if dev_count > 1:
                ret = train_exe.run(feed=feed_dict, fetch_list=metric.vars)
                ret = [[np.mean(v)] for v in ret]
            else:
                ret = train_exe.run(train_prog,
                                    feed=feed_dict,
                                    fetch_list=metric.vars)

            ret = metric.parse(ret)
            if global_step % train_log_step == 0:
                for key, value in ret.items():
                    writer.add_scalar(
                        'train_' + key, value, global_step)

            global_step += 1
            if global_step % eval_step == 0:
                eval_ret = evaluate(model, exe, valid_ds, valid_prog, 1,
                                    evaluator, "valid")

                test_eval_ret = evaluate(model, exe, test_ds, valid_prog, 1,
                                         evaluator, "test")

                eval_ret.update(test_eval_ret)

                sys.stderr.write(json.dumps(eval_ret, indent=4) + "\n")

                for key, value in eval_ret.items():
                    writer.add_scalar(key, value, global_step)

                if eval_ret["valid_hits@100"] > best_model:
                    F.io.save_persistables(
                        exe,
                        os.path.join(output_path, "checkpoint"), train_prog)
                    eval_ret["step"] = global_step
                    with open(os.path.join(output_path, "best.txt"), "w") as f:
                        f.write(json.dumps(eval_ret, indent=2) + '\n')
                    best_model = eval_ret["valid_hits@100"]
        # Epoch End
        eval_ret = evaluate(model, exe, valid_ds, valid_prog, 1, evaluator,
                            "valid")

        test_eval_ret = evaluate(model, exe, test_ds, valid_prog, 1, evaluator,
                                 "test")

        eval_ret.update(test_eval_ret)
        sys.stderr.write(json.dumps(eval_ret, indent=4) + "\n")

        for key, value in eval_ret.items():
            writer.add_scalar(key, value, global_step)

        if eval_ret["valid_hits@100"] > best_model:
            F.io.save_persistables(exe,
                                   os.path.join(output_path, "checkpoint"),
                                   train_prog)
            eval_ret["step"] = global_step
            with open(os.path.join(output_path, "best.txt"), "w") as f:
                f.write(json.dumps(eval_ret, indent=2) + '\n')
            best_model = eval_ret["valid_hits@100"]

    writer.close()
