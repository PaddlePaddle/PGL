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
"""
This file provides the multi class task for testing the embedding 
learned by LINE model.
"""
import argparse
import time
import math
import os
import random

import numpy as np
import sklearn.metrics
from sklearn.metrics import f1_score

import pgl
from pgl.utils import op
import paddle.fluid as fluid
import paddle.fluid.layers as l
from pgl.utils.logger import log
from data_loader import FlickrDataset


def load_param(dirname, var_name_list):
    """load_param"""
    for var_name in var_name_list:
        var = fluid.global_scope().find_var(var_name)
        var_tensor = var.get_tensor()
        var_tmp = np.load(os.path.join(dirname, var_name + '.npy'))
        var_tensor.set(var_tmp, fluid.CPUPlace())


def set_seed(seed):
    """Set global random seed.
    """
    random.seed(seed)
    np.random.seed(seed)


def node_classify_model(graph,
                        num_labels,
                        embed_dim=16,
                        name='node_classify_task'):
    """Build node classify model.

    Args:
        graph: The :code:`Graph` data object.

        num_labels: The number of labels.

        embed_dim: The dimension of embedding.

        name: The name of the model.
    """
    pyreader = l.py_reader(
        capacity=70,
        shapes=[[-1, 1], [-1, num_labels]],
        dtypes=['int64', 'float32'],
        lod_levels=[0, 0],
        name=name + '_pyreader',
        use_double_buffer=True)
    nodes, labels = l.read_file(pyreader)
    embed_nodes = l.embedding(
        input=nodes, size=[graph.num_nodes, embed_dim], param_attr='shared_w')
    embed_nodes.stop_gradient = True
    logits = l.fc(input=embed_nodes, size=num_labels)
    loss = l.sigmoid_cross_entropy_with_logits(logits, labels)
    loss = l.reduce_mean(loss)
    prob = l.sigmoid(logits)
    topk = l.reduce_sum(labels, -1)
    return {
        'pyreader': pyreader,
        'loss': loss,
        'prob': prob,
        'labels': labels,
        'topk': topk
    }
    #  return pyreader, loss, prob, labels, topk


def node_classify_generator(graph,
                            all_nodes=None,
                            batch_size=512,
                            epoch=1,
                            shuffle=True):
    """Data generator for node classify.

    Args:
        graph: The :code:`Graph` data object.

        all_nodes: the total number of nodes.

        batch_size: batch size for training.

        epoch: The number of epochs.

        shuffle: Random shuffle of data.
    """

    if all_nodes is None:
        all_nodes = np.arange(graph.num_nodes)

    def batch_nodes_generator(shuffle=shuffle):
        """Batch nodes generator.
        """
        perm = np.arange(len(all_nodes), dtype=np.int64)
        if shuffle:
            np.random.shuffle(perm)
        start = 0
        while start < len(all_nodes):
            yield all_nodes[perm[start:start + batch_size]]
            start += batch_size

    def wrapper():
        """Wrapper function.
        """
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
    """Calculate top K F1 score.
    """
    assert topk_list is not None or threshold is not None, "one of topklist and threshold should not be None"
    if threshold is not None:
        preds = probs > threshold
    else:
        preds = np.zeros_like(labels, dtype=np.int64)
        for idx, (prob, topk) in enumerate(zip(np.argsort(probs), topk_list)):
            preds[idx][prob[-int(topk):]] = 1
    return f1_score(labels, preds, average=average)


def main(args):
    """The main funciton for nodes classify task.
    """
    set_seed(args.seed)
    log.info(args)
    dataset = FlickrDataset(args.data_path, train_percentage=args.percent)

    train_steps = (len(dataset.train_index) // args.batch_size) * args.epochs
    place = fluid.CUDAPlace(0) if args.use_cuda else fluid.CPUPlace()
    train_prog = fluid.Program()
    test_prog = fluid.Program()
    startup_prog = fluid.Program()

    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            train_model = node_classify_model(
                dataset.graph,
                dataset.num_groups,
                embed_dim=args.embed_dim,
                name='train')

            lr = l.polynomial_decay(args.lr, train_steps, 0.0001)
            adam = fluid.optimizer.Adam(lr)
            adam.minimize(train_model['loss'])
    with fluid.program_guard(test_prog, startup_prog):
        with fluid.unique_name.guard():
            test_model = node_classify_model(
                dataset.graph,
                dataset.num_groups,
                embed_dim=args.embed_dim,
                name='test')
    test_prog = test_prog.clone(for_test=True)
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    train_model['pyreader'].decorate_tensor_provider(
        node_classify_generator(
            dataset.graph,
            dataset.train_index,
            batch_size=args.batch_size,
            epoch=args.epochs))
    test_model['pyreader'].decorate_tensor_provider(
        node_classify_generator(
            dataset.graph,
            dataset.test_index,
            batch_size=args.batch_size,
            epoch=1))

    def existed_params(var):
        """existed_params
        """
        if not isinstance(var, fluid.framework.Parameter):
            return False
        return os.path.exists(os.path.join(args.ckpt_path, var.name))

    log.info('loading pretrained parameters from npy')
    load_param(args.ckpt_path, ['shared_w'])

    step = 0
    prev_time = time.time()
    train_model['pyreader'].start()

    final_macro_f1 = 0.0
    final_micro_f1 = 0.0
    while 1:
        try:
            train_loss_val, train_probs_val, train_labels_val, train_topk_val = exe.run(
                train_prog,
                fetch_list=[
                    train_model['loss'], train_model['prob'],
                    train_model['labels'], train_model['topk']
                ],
                return_numpy=True)
            train_macro_f1 = topk_f1_score(train_labels_val, train_probs_val,
                                           train_topk_val, "macro",
                                           args.threshold)
            train_micro_f1 = topk_f1_score(train_labels_val, train_probs_val,
                                           train_topk_val, "micro",
                                           args.threshold)
            step += 1
            log.info("Step %d " % step + "Train Loss: %f " % train_loss_val +
                     "Train Macro F1: %f " % train_macro_f1 +
                     "Train Micro F1: %f " % train_micro_f1)
        except fluid.core.EOFException:
            train_model['pyreader'].reset()
            break

        test_model['pyreader'].start()
        test_probs_vals, test_labels_vals, test_topk_vals = [], [], []
        while 1:
            try:
                test_loss_val, test_probs_val, test_labels_val, test_topk_val = exe.run(
                    test_prog,
                    fetch_list=[
                        test_model['loss'], test_model['prob'],
                        test_model['labels'], test_model['topk']
                    ],
                    return_numpy=True)
                test_probs_vals.append(
                    test_probs_val), test_labels_vals.append(test_labels_val)
                test_topk_vals.append(test_topk_val)
            except fluid.core.EOFException:
                test_model['pyreader'].reset()
                test_probs_array = np.concatenate(test_probs_vals)
                test_labels_array = np.concatenate(test_labels_vals)
                test_topk_array = np.concatenate(test_topk_vals)
                test_macro_f1 = topk_f1_score(
                    test_labels_array, test_probs_array, test_topk_array,
                    "macro", args.threshold)
                test_micro_f1 = topk_f1_score(
                    test_labels_array, test_probs_array, test_topk_array,
                    "micro", args.threshold)
                log.info("\t\tStep %d " % step + "Test Loss: %f " %
                         test_loss_val + "Test Macro F1: %f " % test_macro_f1 +
                         "Test Micro F1: %f " % test_micro_f1)
                final_macro_f1 = max(test_macro_f1, final_macro_f1)
                final_micro_f1 = max(test_micro_f1, final_micro_f1)
                break

    log.info("\nFinal test Macro F1: %f " % final_macro_f1 +
             "Final test Micro F1: %f " % final_micro_f1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LINE')
    parser.add_argument(
        '--data_path',
        type=str,
        default='./data/flickr/',
        help='dataset for training')
    parser.add_argument("--use_cuda", action='store_true', help="use_cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1667)
    parser.add_argument(
        "--lr", type=float, default=0.025, help='learning rate')
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--percent",
        type=float,
        default=0.5,
        help="the percentage of data as training data")
    parser.add_argument(
        "--ckpt_path", type=str, default="./checkpoints/model/model_epoch_0/")
    args = parser.parse_args()
    main(args)
