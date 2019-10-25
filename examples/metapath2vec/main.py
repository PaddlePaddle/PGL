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
This file implement the training process of metapath2vec model.
"""
import os
import sys
import argparse
import time
import numpy as np
import logging
import pickle as pkl
import shutil
import glob

import pgl
from pgl.utils import paddle_helper
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as fl

from utils import *
import Dataset
import model as Models
from pgl.utils import mp_reader
from sklearn.metrics import (auc, f1_score, precision_recall_curve,
                             roc_auc_score)


def set_seed(seed):
    """Set global random seed."""
    random.seed(seed)
    np.random.seed(seed)


def save_param(dirname, var_name_list):
    """save_param"""
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    for var_name in var_name_list:
        var = fluid.global_scope().find_var(var_name)
        var_tensor = var.get_tensor()
        np.save(os.path.join(dirname, var_name + '.npy'), np.array(var_tensor))


def multiprocess_data_generator(config, dataset):
    """Using multiprocess to generate training data.
    """
    num_sample_workers = config['trainer']['args']['num_sample_workers']

    walkpath_files = [[] for i in range(num_sample_workers)]
    for idx, f in enumerate(glob.glob(dataset.walk_files)):
        walkpath_files[idx % num_sample_workers].append(f)

    gen_data_pool = [
        dataset.pairs_generator(files) for files in walkpath_files
    ]
    if num_sample_workers == 1:
        gen_data_func = gen_data_pool[0]
    else:
        gen_data_func = mp_reader.multiprocess_reader(
            gen_data_pool, use_pipe=True, queue_size=100)

    return gen_data_func


def run_epoch(epoch,
              config,
              data_generator,
              train_prog,
              model,
              feed_dict,
              exe,
              for_test=False):
    """Run training process of every epoch.
    """
    total_loss = []
    for idx, batch_data in enumerate(data_generator()):
        feed_dict['train_inputs'] = batch_data['src']
        feed_dict['train_labels'] = batch_data['pos']
        feed_dict['train_negs'] = batch_data['negs']

        loss, lr = exe.run(train_prog,
                           feed=feed_dict,
                           fetch_list=[model.loss, model.lr],
                           return_numpy=True)
        total_loss.append(loss[0])

        if (idx + 1) % 500 == 0:
            avg_loss = np.mean(total_loss)
            logging.info("epoch %d | step %d | lr %.4f | train_loss %f " %
                         (epoch, idx + 1, lr, avg_loss))
            total_loss = []


def main(config):
    """main function for training metapath2vec model.
    """
    logging.info(config)

    set_seed(config['seed'])

    dataset = getattr(
        Dataset, config['data_loader']['type'])(config['data_loader']['args'])
    data_generator = multiprocess_data_generator(config, dataset)

    # move word2id file to checkpoints directory
    src_word2id_file = dataset.word2id_file
    dst_wor2id_file = config['trainer']['args']['save_dir'] + config[
        'data_loader']['args']['word2id_file']
    logging.info('backup word2id file to %s' % dst_wor2id_file)
    shutil.move(src_word2id_file, dst_wor2id_file)

    place = fluid.CUDAPlace(0) if config['use_cuda'] else fluid.CPUPlace()
    train_program = fluid.Program()
    startup_program = fluid.Program()

    with fluid.program_guard(train_program, startup_program):
        model = getattr(Models, config['model']['type'])(
            dataset=dataset, config=config['model']['args'], place=place)

    with fluid.program_guard(train_program, startup_program):
        global_steps = int(dataset.sentences_count *
                           config['trainer']['args']['epochs'] /
                           config['data_loader']['args']['batch_size'])
        model.backward(global_steps, config['optimizer']['args'])

    # train
    exe = fluid.Executor(place)
    exe.run(startup_program)
    feed_dict = {}

    logging.info('training...')
    for epoch in range(1, 1 + config['trainer']['args']['epochs']):
        run_epoch(epoch, config['trainer']['args'], data_generator,
                  train_program, model, feed_dict, exe)

        logging.info('saving model...')
        cur_save_path = os.path.join(config['trainer']['args']['save_dir'],
                                     "model_epoch%d" % (epoch))
        save_param(cur_save_path, ['content'])

    logging.info('finishing training')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='metapath2vec')
    parser.add_argument(
        '-c',
        '--config',
        default=None,
        type=str,
        help='config file path (default: None)')
    parser.add_argument(
        '-n',
        '--taskname',
        default=None,
        type=str,
        help='task name(default: None)')
    args = parser.parse_args()

    if args.config:
        # load config file
        config = Config(args.config, isCreate=True, isSave=True)
        config = config()
    else:
        raise AssertionError(
            "Configuration file need to be specified. Add '-c config.yaml', for example."
        )

    log_format = '%(asctime)s-%(levelname)s-%(name)s: %(message)s'
    logging.basicConfig(
        level=getattr(logging, config['log_level'].upper()), format=log_format)

    main(config)
