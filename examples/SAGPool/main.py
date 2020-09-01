# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
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

import sys
import os
import argparse
import pgl
from pgl.utils.logger import log
import paddle

import re
import time
import random
import numpy as np
import math

import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as L
import pgl
from pgl.utils.logger import log

from model import GlobalModel
from base_dataset import Subset, Dataset
from dataloader import GraphDataloader
from args import parser
import warnings
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
     
def main(args, train_dataset, val_dataset, test_dataset):
    """main function for running one testing results.
    """
    log.info("Train Examples: %s" % len(train_dataset))
    log.info("Val Examples: %s" % len(val_dataset))
    log.info("Test Examples: %s" % len(test_dataset))

    train_program = fluid.Program()
    train_program.random_seed = args.seed
    startup_program = fluid.Program()
    startup_program.random_seed = args.seed

    if args.use_cuda:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    log.info("building model")

    with fluid.program_guard(train_program, startup_program):
        with fluid.unique_name.guard():
            graph_model = GlobalModel(args, dataset) 
            train_loader = GraphDataloader(train_dataset,
                                           graph_model.graph_wrapper,
                                           batch_size=args.batch_size)
            optimizer = fluid.optimizer.Adam(learning_rate=args.learning_rate,
                regularization=fluid.regularizer.L2DecayRegularizer(args.weight_decay))
            optimizer.minimize(graph_model.loss)

    exe.run(startup_program)
    test_program = fluid.Program()
    test_program = train_program.clone(for_test=True)

    val_loader = GraphDataloader(val_dataset,   
                                 graph_model.graph_wrapper,
                                 batch_size=args.batch_size,
                                 shuffle=False)
    test_loader = GraphDataloader(test_dataset,
                                  graph_model.graph_wrapper,
                                  batch_size=args.batch_size,
                                  shuffle=False)

    min_loss = 1e10
    global_step = 0
    for epoch in range(args.epochs):
        for feed_dict in train_loader:
            loss, pred = exe.run(train_program,
                           feed=feed_dict,
                           fetch_list=[graph_model.loss, graph_model.pred])

            log.info("Epoch: %d, global_step: %d, Training loss: %f" \
                     % (epoch, global_step, loss))
            global_step += 1

        # validation
        valid_loss = 0.
        correct = 0.
        for feed_dict in val_loader: 
            valid_loss_, correct_ = exe.run(test_program,
                                 feed=feed_dict,
                                 fetch_list=[graph_model.loss, graph_model.correct])
            valid_loss += valid_loss_
            correct += correct_ 

        if epoch % 50 == 0:
            log.info("Epoch:%d, Validation loss: %f, Validation acc: %f" \
                    % (epoch, valid_loss, correct / len(val_loader)))

        if valid_loss < min_loss:
            min_loss = valid_loss
            patience = 0
            path = "./save/%s" % args.dataset_name
            if not os.path.exists(path):
                os.makedirs(path)
            fluid.save(train_program, "%s/%s" \
                       % (path, args.save_model))
            log.info("Model saved at epoch %d" % epoch)
        else:
            patience += 1
        if patience > args.patience:
            break

    correct = 0.
    fluid.load(test_program, "./save/%s/%s" \
               % (args.dataset_name, args.save_model), exe)
    for feed_dict in test_loader:
        correct_ = exe.run(test_program,
                           feed=feed_dict,
                           fetch_list=[graph_model.correct])
        correct += correct_[0]
    log.info("Test acc: %f" % (correct / len(test_loader)))
    return correct / len(test_loader)
    

def split_10_cv(dataset, args):
    """10 folds cross validation
    """
    dataset.shuffle()
    X = np.array([0] * len(dataset))
    y = X
    kf = KFold(n_splits=10, shuffle=False)

    i = 1
    test_acc = []
    for train_index, test_index in kf.split(X, y):
        train_val_dataset = Subset(dataset, train_index)
        test_dataset = Subset(dataset, test_index)
        train_val_index_range = list(range(0, len(train_val_dataset)))
        num_val = int(len(train_val_dataset) / 9)
        val_dataset = Subset(train_val_dataset, train_val_index_range[:num_val])
        train_dataset = Subset(train_val_dataset, train_val_index_range[num_val:])

        log.info("######%d fold of 10-fold cross validation######" % i)
        i += 1
        test_acc_ = main(args, train_dataset, val_dataset, test_dataset)
        test_acc.append(test_acc_)

    mean_acc = sum(test_acc) / len(test_acc)    
    return mean_acc, test_acc


def random_seed_20(args, dataset):
    """run for 20 random seeds
    """
    alist = random.sample(range(1,1000),20)
    test_acc_fold = []
    for seed in alist:
        log.info('############ Seed %d ############' % seed)
        args.seed = seed

        test_acc_fold_, _ = split_10_cv(dataset, args)
        log.info('Mean test acc at seed %d: %f' % (seed, test_acc_fold_))
        test_acc_fold.append(test_acc_fold_)

    mean_acc = sum(test_acc_fold) / len(test_acc_fold)
    temp = [(acc - mean_acc) * (acc - mean_acc) for acc in test_acc_fold]
    standard_std = math.sqrt(sum(temp) / len(test_acc_fold))

    log.info('Final mean test acc using 20 random seeds(mean for 10-fold): %f' % (mean_acc))
    log.info('Final standard std using 20 random seeds(mean for 10-fold): %f' % (standard_std))

    
if __name__ == "__main__":
    args = parser.parse_args()
    log.info('loading data...')
    dataset = Dataset(args)
    log.info("preprocess finish.")
    args.num_classes = dataset.num_classes
    args.num_features = dataset.num_features
    random_seed_20(args, dataset)
