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
from sklearn import metrics

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
    elif name == "ArXiv":
        dataset = data_loader.ArXivDataset()
    else:
        raise ValueError(name + " dataset doesn't exists")
    return dataset


def binary_op(u_embed, v_embed, binary_op_type):
    if binary_op_type == "Average":
        edge_embed = (u_embed + v_embed) / 2
    elif binary_op_type == "Hadamard":
        edge_embed = u_embed * v_embed
    elif binary_op_type == "Weighted-L1":
        edge_embed = l.abs(u_embed - v_embed)
    elif binary_op_type == "Weighted-L2":
        edge_embed = (u_embed - v_embed) * (u_embed - v_embed)
    else:
        raise ValueError(binary_op_type + " binary_op_type doesn't exists")
    return edge_embed


def link_predict_model(num_nodes,
                       hidden_size=16,
                       name='link_predict_task',
                       binary_op_type="Weighted-L2"):
    pyreader = l.py_reader(
        capacity=70,
        shapes=[[-1, 1], [-1, 1], [-1, 1]],
        dtypes=['int64', 'int64', 'int64'],
        lod_levels=[0, 0, 0],
        name=name + '_pyreader',
        use_double_buffer=True)
    u, v, label = l.read_file(pyreader)
    u_embed = l.embedding(
        input=u,
        size=[num_nodes, hidden_size],
        param_attr=fluid.ParamAttr(name='content'))
    v_embed = l.embedding(
        input=v,
        size=[num_nodes, hidden_size],
        param_attr=fluid.ParamAttr(name='content'))
    u_embed.stop_gradient = True
    v_embed.stop_gradient = True

    edge_embed = binary_op(u_embed, v_embed, binary_op_type)
    logit = l.fc(input=edge_embed, size=1)
    loss = l.sigmoid_cross_entropy_with_logits(logit, l.cast(label, 'float32'))
    loss = l.reduce_mean(loss)

    prob = l.sigmoid(logit)
    return pyreader, loss, prob, label


def link_predict_generator(pos_edges,
                           neg_edges,
                           batch_size=512,
                           epoch=2000,
                           shuffle=True):

    all_edges = []
    for (u, v) in pos_edges:
        all_edges.append([u, v, 1])
    for (u, v) in neg_edges:
        all_edges.append([u, v, 0])
    all_edges = np.array(all_edges, np.int64)

    def batch_edges_generator(shuffle=shuffle):
        perm = np.arange(len(all_edges), dtype=np.int64)
        if shuffle:
            np.random.shuffle(perm)
        start = 0
        while start < len(all_edges):
            yield all_edges[perm[start:start + batch_size]]
            start += batch_size

    def wrapper():
        for _ in range(epoch):
            for batch_edges in batch_edges_generator():
                yield batch_edges.T[0:1].T, batch_edges.T[
                    1:2].T, batch_edges.T[2:3].T

    return wrapper


def main(args):
    hidden_size = args.hidden_size
    epoch = args.epoch
    ckpt_path = args.ckpt_path

    dataset = load(args.dataset)

    num_edges = len(dataset.pos_edges) + len(dataset.neg_edges)
    train_num_edges = int(len(dataset.pos_edges) * 0.5) + int(
        len(dataset.neg_edges) * 0.5)
    test_num_edges = num_edges - train_num_edges

    train_steps = (train_num_edges // train_num_edges) * epoch

    place = fluid.CUDAPlace(0) if args.use_cuda else fluid.CPUPlace()
    train_prog = fluid.Program()
    test_prog = fluid.Program()
    startup_prog = fluid.Program()

    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            train_pyreader, train_loss, train_probs, train_labels = link_predict_model(
                dataset.graph.num_nodes, hidden_size=hidden_size, name='train')
            lr = l.polynomial_decay(0.025, train_steps, 0.0001)
            adam = fluid.optimizer.Adam(lr)
            adam.minimize(train_loss)

    with fluid.program_guard(test_prog, startup_prog):
        with fluid.unique_name.guard():
            test_pyreader, test_loss, test_probs, test_labels = link_predict_model(
                dataset.graph.num_nodes, hidden_size=hidden_size, name='test')
    test_prog = test_prog.clone(for_test=True)

    train_pyreader.decorate_tensor_provider(
        link_predict_generator(
            dataset.pos_edges[:train_num_edges // 2],
            dataset.neg_edges[:train_num_edges // 2],
            batch_size=train_num_edges,
            epoch=epoch))

    test_pyreader.decorate_tensor_provider(
        link_predict_generator(
            dataset.pos_edges[train_num_edges // 2:],
            dataset.neg_edges[train_num_edges // 2:],
            batch_size=test_num_edges,
            epoch=1))

    exe = fluid.Executor(place)
    exe.run(startup_prog)
    train_pyreader.start()

    def existed_params(var):
        if not isinstance(var, fluid.framework.Parameter):
            return False
        return os.path.exists(os.path.join(ckpt_path, var.name))

    fluid.io.load_vars(
        exe, ckpt_path, main_program=train_prog, predicate=existed_params)
    step = 0
    prev_time = time.time()

    while 1:
        try:
            train_loss_val, train_probs_val, train_labels_val = exe.run(
                train_prog,
                fetch_list=[train_loss, train_probs, train_labels],
                return_numpy=True)
            fpr, tpr, thresholds = metrics.roc_curve(train_labels_val,
                                                     train_probs_val)
            train_auc = metrics.auc(fpr, tpr)
            step += 1
            log.info("Step %d " % step + "Train Loss: %f " % train_loss_val +
                     "Train AUC: %f " % train_auc)
        except fluid.core.EOFException:
            train_pyreader.reset()
            break

        test_pyreader.start()
        test_probs_vals, test_labels_vals = [], []
        while 1:
            try:
                test_loss_val, test_probs_val, test_labels_val = exe.run(
                    test_prog,
                    fetch_list=[test_loss, test_probs, test_labels],
                    return_numpy=True)
                test_probs_vals.append(
                    test_probs_val), test_labels_vals.append(test_labels_val)
            except fluid.core.EOFException:
                test_pyreader.reset()
                test_probs_array = np.concatenate(test_probs_vals)
                test_labels_array = np.concatenate(test_labels_vals)
                fpr, tpr, thresholds = metrics.roc_curve(test_labels_array,
                                                         test_probs_array)
                test_auc = metrics.auc(fpr, tpr)
                log.info("\t\tStep %d " % step + "Test Loss: %f " %
                         test_loss_val + "Test AUC: %f " % test_auc)
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='deepwalk')
    parser.add_argument(
        "--dataset",
        type=str,
        default="ArXiv",
        help="dataset (BlogCatalog, ArXiv)")
    parser.add_argument("--use_cuda", action='store_true', help="use_cuda")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--epoch", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument(
        "--ckpt_path", type=str, default="./tmp/deepwalk_arxiv/paddle_model")
    args = parser.parse_args()
    log.info(args)
    main(args)
