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
This file provides the multi class task for testing the embedding learned by metapath2vec model.
"""
import argparse
import sys
import os
import tqdm
import time
import math
import logging
import random
import pickle as pkl

import numpy as np
import sklearn.metrics
from sklearn.metrics import f1_score

import pgl
import paddle.fluid as fluid
import paddle.fluid.layers as fl


def load_data(file_):
    """Load data for node classification.
    """
    words_label = []
    line_count = 0
    with open(file_, 'r') as reader:
        for line in reader:
            line_count += 1
            tokens = line.strip().split()
            word, label = int(tokens[0]), int(tokens[1]) - 1
            words_label.append((word, label))

    words_label = np.array(words_label, dtype=np.int64)
    np.random.shuffle(words_label)

    logging.info('%d/%d word_label pairs have been loaded' %
                 (len(words_label), line_count))
    return words_label


def node_classify_model(config):
    """Build node classify model.
    """
    nodes = fl.data('nodes', shape=[None, 1], dtype='int64')
    labels = fl.data('labels', shape=[None, 1], dtype='int64')

    embed_nodes = fl.embedding(
        input=nodes,
        size=[config.num_nodes, config.embed_dim],
        param_attr=fluid.ParamAttr(name='weight'))

    embed_nodes.stop_gradient = True
    probs = fl.fc(input=embed_nodes, size=config.num_labels, act='softmax')
    predict = fl.argmax(probs, axis=-1)
    loss = fl.cross_entropy(input=probs, label=labels)
    loss = fl.reduce_mean(loss)

    return {
        'loss': loss,
        'probs': probs,
        'predict': predict,
        'labels': labels,
    }


def run_epoch(exe, prog, model, feed_dict, lr):
    """Run training process of every epoch.
    """
    if lr is None:
        loss, predict = exe.run(prog,
                                feed=feed_dict,
                                fetch_list=[model['loss'], model['predict']],
                                return_numpy=True)
        lr_ = 0
    else:
        loss, predict, lr_ = exe.run(
            prog,
            feed=feed_dict,
            fetch_list=[model['loss'], model['predict'], lr],
            return_numpy=True)

    macro_f1 = f1_score(feed_dict['labels'], predict, average="macro")
    micro_f1 = f1_score(feed_dict['labels'], predict, average="micro")

    return {
        'loss': loss,
        'pred': predict,
        'lr': lr_,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1
    }


def main(args):
    """main function for training node classification task.
    """
    words_label = load_data(args.dataset)

    # split data for training and testing
    split_position = int(words_label.shape[0] * args.train_percent)
    train_words_label = words_label[0:split_position, :]
    test_words_label = words_label[split_position:, :]

    place = fluid.CUDAPlace(0) if args.use_cuda else fluid.CPUPlace()
    train_prog = fluid.Program()
    test_prog = fluid.Program()
    startup_prog = fluid.Program()

    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            model = node_classify_model(args)

    test_prog = train_prog.clone(for_test=True)

    with fluid.program_guard(train_prog, startup_prog):
        lr = fl.polynomial_decay(args.lr, 1000, 0.001)
        adam = fluid.optimizer.Adam(lr)
        adam.minimize(model['loss'])

    exe = fluid.Executor(place)
    exe.run(startup_prog)

    def existed_params(var):
        if not isinstance(var, fluid.framework.Parameter):
            return False
        return os.path.exists(os.path.join(args.ckpt_path, var.name))

    fluid.io.load_vars(
        exe, args.ckpt_path, main_program=train_prog, predicate=existed_params)
    #  load_param(args.ckpt_path, ['content'])

    feed_dict = {}
    X = train_words_label[:, 0].reshape(-1, 1)
    labels = train_words_label[:, 1].reshape(-1, 1)
    logging.info('%d/%d data to train' %
                 (labels.shape[0], words_label.shape[0]))

    test_feed_dict = {}
    test_X = test_words_label[:, 0].reshape(-1, 1)
    test_labels = test_words_label[:, 1].reshape(-1, 1)
    logging.info('%d/%d data to test' %
                 (test_labels.shape[0], words_label.shape[0]))

    for epoch in range(args.epochs):
        feed_dict['nodes'] = X
        feed_dict['labels'] = labels
        train_result = run_epoch(exe, train_prog, model, feed_dict, lr)

        test_feed_dict['nodes'] = test_X
        test_feed_dict['labels'] = test_labels

        test_result = run_epoch(exe, test_prog, model, test_feed_dict, lr=None)

        logging.info(
            'epoch %d | lr %.4f | train_loss %.5f | train_macro_F1 %.4f | train_micro_F1 %.4f | test_loss %.5f | test_macro_F1 %.4f | test_micro_F1 %.4f'
            % (epoch, train_result['lr'], train_result['loss'],
               train_result['macro_f1'], train_result['micro_f1'],
               test_result['loss'], test_result['macro_f1'],
               test_result['micro_f1']))

    logging.info(
        'final_test_macro_f1 score: %.4f | final_test_micro_f1 score: %.4f' %
        (test_result['macro_f1'], test_result['micro_f1']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='multi_class')
    parser.add_argument(
        '--dataset',
        default=None,
        type=str,
        help='training and testing data file(default: None)')
    parser.add_argument(
        '--ckpt_path', default=None, type=str, help='task name(default: None)')
    parser.add_argument("--use_cuda", action='store_true', help="use_cuda")
    parser.add_argument(
        '--train_percent',
        default=0.5,
        type=float,
        help='train_percent(default: 0.5)')
    parser.add_argument(
        '--num_labels',
        default=4,
        type=int,
        help='number of labels(default: 4)')
    parser.add_argument(
        '--epochs',
        default=100,
        type=int,
        help='number of epochs for training(default: 100)')
    parser.add_argument(
        '--lr',
        default=0.025,
        type=float,
        help='learning rate(default: 0.025)')
    parser.add_argument(
        '--num_nodes', default=0, type=int, help='number of nodes')
    parser.add_argument(
        '--embed_dim',
        default=64,
        type=int,
        help='dimension of embedding(default: 64)')
    args = parser.parse_args()

    log_format = '%(asctime)s-%(levelname)s-%(name)s: %(message)s'
    logging.basicConfig(level='INFO', format=log_format)

    main(args)
