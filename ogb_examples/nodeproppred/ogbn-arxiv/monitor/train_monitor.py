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
from tensorboardX import SummaryWriter
from ogb.nodeproppred import Evaluator
from ogb.nodeproppred import NodePropPredDataset


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
        d_name = "ogbn-arxiv"
        dataset = NodePropPredDataset(name=d_name)
        graph, label = dataset[0]
        self.num_nodes = graph["num_nodes"]
        self.ogb_evaluator = Evaluator(name="ogbn-arxiv")

    def eval(self, scores, labels, phase):
        pred = (np.argmax(scores, axis=1)).reshape([-1, 1])
        ret = {}
        ret['%s_acc' % (phase)] = self.ogb_evaluator.eval({
            'y_true': labels,
            'y_pred': pred,
        })['acc']
        return ret


def evaluate(model, valid_exe, valid_ds, valid_prog, dev_count, evaluator,
             phase, full_batch):
    """evaluate """
    cc = 0
    scores = []
    labels = []
    if full_batch:
        valid_iter = _full_batch_wapper(valid_ds)
    else:
        valid_iter = valid_ds.generator

    for feed_dict in tqdm.tqdm(
            multi_device(valid_iter(), dev_count), desc='evaluating'):
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


def _full_batch_wapper(ds):
    feed_dict = {}
    feed_dict["batch_nodes"] = np.array(ds.nodes_idx, dtype="int64")
    feed_dict["labels"] = np.array(ds.labels, dtype="int64")

    def r():
        yield feed_dict

    return r


def train_and_evaluate(exe,
                       train_exe,
                       valid_exe,
                       train_ds,
                       valid_ds,
                       test_ds,
                       train_prog,
                       valid_prog,
                       full_batch,
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

    writer = SummaryWriter(log_path)

    best_model = 0

    if full_batch:
        train_iter = _full_batch_wapper(train_ds)
    else:
        train_iter = train_ds.generator

    for e in range(epoch):
        ret_sum_loss = 0
        per_step = 0
        scores = []
        labels = []
        for feed_dict in tqdm.tqdm(
                multi_device(train_iter(), dev_count), desc='Epoch %s' % e):
            if dev_count > 1:
                ret = train_exe.run(feed=feed_dict, fetch_list=metric.vars)
                ret = [[np.mean(v)] for v in ret]
            else:
                ret = train_exe.run(
                    train_prog,
                    feed=feed_dict,
                    fetch_list=[model.loss, model.logits, model.labels]
                    #fetch_list=metric.vars
                )
            scores.append(ret[1])
            labels.append(ret[2])
            ret = [ret[0]]

            ret = metric.parse(ret)
            if global_step % train_log_step == 0:
                for key, value in ret.items():
                    writer.add_scalar(
                        'train_' + key, value, global_step=global_step)
            ret_sum_loss += ret['loss']
            per_step += 1
            global_step += 1
            if global_step % eval_step == 0:
                eval_ret = evaluate(model, exe, valid_ds, valid_prog, 1,
                                    evaluator, "valid", full_batch)
                test_eval_ret = evaluate(model, exe, test_ds, valid_prog, 1,
                                         evaluator, "test", full_batch)
                eval_ret.update(test_eval_ret)
                sys.stderr.write(json.dumps(eval_ret, indent=4) + "\n")
                for key, value in eval_ret.items():
                    writer.add_scalar(key, value, global_step=global_step)
                if eval_ret["valid_acc"] > best_model:
                    F.io.save_persistables(
                        exe,
                        os.path.join(output_path, "checkpoint"), train_prog)
                    eval_ret["epoch"] = e
                    #eval_ret["step"] = global_step
                    with open(os.path.join(output_path, "best.txt"), "w") as f:
                        f.write(json.dumps(eval_ret, indent=2) + '\n')
                    best_model = eval_ret["valid_acc"]
        scores = np.vstack(scores)
        labels = np.vstack(labels)

        ret = evaluator.eval(scores, labels, "train")
        sys.stderr.write(json.dumps(ret, indent=4) + "\n")
        #print(json.dumps(ret, indent=4) + "\n")
        # Epoch End
        sys.stderr.write("epoch:{}, average loss {}\n".format(e, ret_sum_loss /
                                                              per_step))
        eval_ret = evaluate(model, exe, valid_ds, valid_prog, 1, evaluator,
                            "valid", full_batch)
        test_eval_ret = evaluate(model, exe, test_ds, valid_prog, 1, evaluator,
                                 "test", full_batch)
        eval_ret.update(test_eval_ret)
        sys.stderr.write(json.dumps(eval_ret, indent=4) + "\n")

        for key, value in eval_ret.items():
            writer.add_scalar(key, value, global_step=global_step)

        if eval_ret["valid_acc"] > best_model:
            F.io.save_persistables(exe,
                                   os.path.join(output_path, "checkpoint"),
                                   train_prog)
            #eval_ret["step"] = global_step
            eval_ret["epoch"] = e
            with open(os.path.join(output_path, "best.txt"), "w") as f:
                f.write(json.dumps(eval_ret, indent=2) + '\n')
            best_model = eval_ret["valid_acc"]

    writer.close()
