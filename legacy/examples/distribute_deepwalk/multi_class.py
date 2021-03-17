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
import argparse
import time
import math
import os

import numpy as np
import sklearn.metrics
from sklearn.metrics import f1_score

import pgl
from pgl import data_loader
from pgl.utils import op
from pgl.utils.logger import log
import paddle.fluid as fluid
import paddle.fluid.layers as l

np.random.seed(123)


def load(name):
    if name == 'BlogCatalog':
        dataset = data_loader.BlogCatalogDataset()
    else:
        raise ValueError(name + " dataset doesn't exists")
    return dataset


def node_classify_model(graph,
                        num_labels,
                        hidden_size=16,
                        name='node_classify_task'):
    pyreader = l.py_reader(
        capacity=70,
        shapes=[[-1, 1], [-1, num_labels]],
        dtypes=['int64', 'float32'],
        lod_levels=[0, 0],
        name=name + '_pyreader',
        use_double_buffer=True)
    nodes, labels = l.read_file(pyreader)
    embed_nodes = l.embedding(
        input=nodes,
        size=[graph.num_nodes, hidden_size],
        param_attr=fluid.ParamAttr(name='weight'))
    embed_nodes.stop_gradient = True
    logits = l.fc(input=embed_nodes, size=num_labels)
    loss = l.sigmoid_cross_entropy_with_logits(logits, labels)
    loss = l.reduce_mean(loss)
    prob = l.sigmoid(logits)
    topk = l.reduce_sum(labels, -1)
    return pyreader, loss, prob, labels, topk


def node_classify_generator(graph,
                            all_nodes=None,
                            batch_size=512,
                            epoch=1,
                            shuffle=True):

    if all_nodes is None:
        all_nodes = np.arange(graph.num_nodes)
    #labels = (np.random.rand(512, 39) > 0.95).astype(np.float32)

    def batch_nodes_generator(shuffle=shuffle):
        perm = np.arange(len(all_nodes), dtype=np.int64)
        if shuffle:
            np.random.shuffle(perm)
        start = 0
        while start < len(all_nodes):
            yield all_nodes[perm[start:start + batch_size]]
            start += batch_size

    def wrapper():
        for _ in range(epoch):
            for batch_nodes in batch_nodes_generator():
                batch_nodes_expanded = np.expand_dims(batch_nodes,
                                                      -1).astype(np.int64)
                batch_labels = graph.node_feat['group_id'][batch_nodes].astype(
                    np.float32)
                yield [batch_nodes_expanded, batch_labels]

    return wrapper


def topk_f1_score(labels,
                  probs,
                  topk_list=None,
                  average="macro",
                  threshold=None):
    assert topk_list is not None or threshold is not None, "one of topklist and threshold should not be None"
    if threshold is not None:
        preds = probs > threshold
    else:
        preds = np.zeros_like(labels, dtype=np.int64)
        for idx, (prob, topk) in enumerate(zip(np.argsort(probs), topk_list)):
            preds[idx][prob[-int(topk):]] = 1
    return f1_score(labels, preds, average=average)


def main(args):
    hidden_size = args.hidden_size
    epoch = args.epoch
    ckpt_path = args.ckpt_path
    threshold = args.threshold

    dataset = load(args.dataset)

    if args.batch_size is None:
        batch_size = len(dataset.train_index)
    else:
        batch_size = args.batch_size

    train_steps = (len(dataset.train_index) // batch_size) * epoch

    place = fluid.CUDAPlace(0) if args.use_cuda else fluid.CPUPlace()
    train_prog = fluid.Program()
    test_prog = fluid.Program()
    startup_prog = fluid.Program()

    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            train_pyreader, train_loss, train_probs, train_labels, train_topk = node_classify_model(
                dataset.graph,
                dataset.num_groups,
                hidden_size=hidden_size,
                name='train')
            lr = l.polynomial_decay(0.025, train_steps, 0.0001)
            adam = fluid.optimizer.Adam(lr)
            adam.minimize(train_loss)

    with fluid.program_guard(test_prog, startup_prog):
        with fluid.unique_name.guard():
            test_pyreader, test_loss, test_probs, test_labels, test_topk = node_classify_model(
                dataset.graph,
                dataset.num_groups,
                hidden_size=hidden_size,
                name='test')
    test_prog = test_prog.clone(for_test=True)

    exe = fluid.Executor(place)
    exe.run(startup_prog)

    train_pyreader.decorate_tensor_provider(
        node_classify_generator(
            dataset.graph,
            dataset.train_index,
            batch_size=batch_size,
            epoch=epoch))
    test_pyreader.decorate_tensor_provider(
        node_classify_generator(
            dataset.graph, dataset.test_index, batch_size=batch_size, epoch=1))

    def existed_params(var):
        if not isinstance(var, fluid.framework.Parameter):
            return False
        return os.path.exists(os.path.join(ckpt_path, var.name))

    fluid.io.load_vars(
        exe, ckpt_path, main_program=train_prog, predicate=existed_params)
    step = 0
    prev_time = time.time()
    train_pyreader.start()

    while 1:
        try:
            train_loss_val, train_probs_val, train_labels_val, train_topk_val = exe.run(
                train_prog,
                fetch_list=[
                    train_loss, train_probs, train_labels, train_topk
                ],
                return_numpy=True)
            train_macro_f1 = topk_f1_score(train_labels_val, train_probs_val,
                                           train_topk_val, "macro", threshold)
            train_micro_f1 = topk_f1_score(train_labels_val, train_probs_val,
                                           train_topk_val, "micro", threshold)
            step += 1
            log.info("Step %d " % step + "Train Loss: %f " % train_loss_val +
                     "Train Macro F1: %f " % train_macro_f1 +
                     "Train Micro F1: %f " % train_micro_f1)
        except fluid.core.EOFException:
            train_pyreader.reset()
            break

        test_pyreader.start()
        test_probs_vals, test_labels_vals, test_topk_vals = [], [], []
        while 1:
            try:
                test_loss_val, test_probs_val, test_labels_val, test_topk_val = exe.run(
                    test_prog,
                    fetch_list=[
                        test_loss, test_probs, test_labels, test_topk
                    ],
                    return_numpy=True)
                test_probs_vals.append(
                    test_probs_val), test_labels_vals.append(test_labels_val)
                test_topk_vals.append(test_topk_val)
            except fluid.core.EOFException:
                test_pyreader.reset()
                test_probs_array = np.concatenate(test_probs_vals)
                test_labels_array = np.concatenate(test_labels_vals)
                test_topk_array = np.concatenate(test_topk_vals)
                test_macro_f1 = topk_f1_score(
                    test_labels_array, test_probs_array, test_topk_array,
                    "macro", threshold)
                test_micro_f1 = topk_f1_score(
                    test_labels_array, test_probs_array, test_topk_array,
                    "micro", threshold)
                log.info("\t\tStep %d " % step + "Test Loss: %f " %
                         test_loss_val + "Test Macro F1: %f " % test_macro_f1 +
                         "Test Micro F1: %f " % test_micro_f1)
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='node2vec')
    parser.add_argument(
        "--dataset",
        type=str,
        default="BlogCatalog",
        help="dataset (BlogCatalog)")
    parser.add_argument("--use_cuda", action='store_true', help="use_cuda")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--epoch", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./tmp/baseline_node2vec/paddle_model")
    args = parser.parse_args()
    log.info(args)
    main(args)
