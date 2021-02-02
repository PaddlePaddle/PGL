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
This file implement the training process of GATNE model.
"""

import os
import argparse
import time
import numpy as np
import logging
import pickle as pkl

import pgl
from pgl.utils import paddle_helper
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as fl

from utils import *
import Dataset
import model as Model
from sklearn.metrics import (auc, f1_score, precision_recall_curve,
                             roc_auc_score)


def set_seed(seed):
    """Set random seed.
    """
    random.seed(seed)
    np.random.seed(seed)


def produce_model(exe, program, dataset, model, feed_dict):
    """Output the learned model parameters for testing.
    """
    edge_types = dataset.edge_types
    num_nodes = dataset.graph[edge_types[0]].num_nodes
    edge_types_count = len(edge_types)
    neg_num = dataset.config['neg_num']

    final_model = {}
    feed_dict['train_inputs'] = np.array(
        [n for n in range(num_nodes)], dtype=np.int64).reshape(-1, )
    feed_dict['train_labels'] = np.array(
        [n for n in range(num_nodes)], dtype=np.int64).reshape(-1, 1, 1)
    feed_dict['train_negs'] = np.tile(feed_dict['train_labels'],
                                      (1, neg_num)).reshape(-1, neg_num, 1)

    for i in range(edge_types_count):
        feed_dict['train_types'] = np.array(
            [i for _ in range(num_nodes)], dtype=np.int64).reshape(-1, 1)
        edge_node_embed = exe.run(program,
                                  feed=feed_dict,
                                  fetch_list=[model.last_node_embed],
                                  return_numpy=True)[0]
        final_model[edge_types[i]] = edge_node_embed

    return final_model


def evaluate(final_model, edge_types, data):
    """Calculate the AUC score, F1 score and PR score of the final model
    """
    edge_types_count = len(edge_types)
    AUC, F1, PR = [], [], []

    true_edge_data_by_type = data[0]
    fake_edge_data_by_type = data[1]

    for i in range(edge_types_count):
        try:
            local_model = final_model[edge_types[i]]
            true_edges = true_edge_data_by_type[edge_types[i]]
            fake_edges = fake_edge_data_by_type[edge_types[i]]
        except Exception as e:
            logging.warn('edge type not exists. %s' % str(e))
            continue
        tmp_auc, tmp_f1, tmp_pr = calculate_score(local_model, true_edges,
                                                  fake_edges)
        AUC.append(tmp_auc)
        F1.append(tmp_f1)
        PR.append(tmp_pr)

    return {'AUC': np.mean(AUC), 'F1': np.mean(F1), 'PR': np.mean(PR)}


def calculate_score(model, true_edges, fake_edges):
    """Calculate the AUC score, F1 score and PR score of specified edge type
    """
    true_list = list()
    prediction_list = list()
    true_num = 0
    for edge in true_edges:
        tmp_score = get_score(model, edge)
        if tmp_score is not None:
            true_list.append(1)
            prediction_list.append(tmp_score)
            true_num += 1

    for edge in fake_edges:
        tmp_score = get_score(model, edge)
        if tmp_score is not None:
            true_list.append(0)
            prediction_list.append(tmp_score)

    sorted_pred = prediction_list[:]
    sorted_pred.sort()
    threshold = sorted_pred[-true_num]

    y_pred = np.zeros(len(prediction_list), dtype=np.int32)
    for i in range(len(prediction_list)):
        if prediction_list[i] >= threshold:
            y_pred[i] = 1

    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    ps, rs, _ = precision_recall_curve(y_true, y_scores)
    return roc_auc_score(y_true, y_scores), f1_score(y_true, y_pred), auc(rs,
                                                                          ps)


def get_score(local_model, edge):
    """Calculate the cosine similarity score between two nodes.
    """
    try:
        vector1 = local_model[edge[0]]
        vector2 = local_model[edge[1]]
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) *
                                           np.linalg.norm(vector2))
    except Exception as e:
        logging.warn('get_score warning: %s' % str(e))
        return None
        pass


def run_epoch(epoch,
              config,
              dataset,
              data,
              train_prog,
              test_prog,
              model,
              feed_dict,
              exe,
              for_test=False):
    """Run training process of every epoch.
    """
    total_loss = []
    for idx, batch_data in enumerate(data):
        feed_dict['train_inputs'] = batch_data[0]
        feed_dict['train_labels'] = batch_data[1]
        feed_dict['train_negs'] = batch_data[2]
        feed_dict['train_types'] = batch_data[3]

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

    return avg_loss


def save_model(program, exe, dataset, model, feed_dict, filename):
    """Save model.
    """
    final_model = produce_model(exe, program, dataset, model, feed_dict)
    logging.info('saving model in %s' % (filename))
    pkl.dump(final_model, open(filename, 'wb'))


def test(program, exe, dataset, model, feed_dict):
    """Testing and validating.
    """
    final_model = produce_model(exe, program, dataset, model, feed_dict)
    valid_result = evaluate(final_model, dataset.edge_types,
                            dataset.valid_data)
    test_result = evaluate(final_model, dataset.edge_types, dataset.test_data)

    logging.info("valid_AUC %.4f | valid_PR %.4f | valid_F1 %.4f" %
                 (valid_result['AUC'], valid_result['PR'], valid_result['F1']))
    logging.info("test_AUC %.4f | test_PR %.4f | test_F1 %.4f" %
                 (test_result['AUC'], test_result['PR'], test_result['F1']))

    return test_result


def main(config):
    """main function for training GATNE model.
    """
    logging.info(config)

    set_seed(config['seed'])

    dataset = getattr(
        Dataset, config['data_loader']['type'])(config['data_loader']['args'])
    edge_types = dataset.graph.edge_types_info()
    logging.info(['total edge types: ', edge_types])

    # train_pairs is a list of tuple: [(src1, dst1, neg, e1), (src2, dst2, neg, e2)]
    # e(int), edge num count, for select which edge embedding 
    train_pairs_file = config['data_loader']['args']['data_path'] + \
                    config['data_loader']['args']['train_pairs_file']
    if os.path.exists(train_pairs_file):
        logging.info('loading train pairs from pkl file %s' % train_pairs_file)
        train_pairs = pkl.load(open(train_pairs_file, 'rb'))
    else:
        logging.info('generating walks')
        all_walks = dataset.generate_walks()
        logging.info('generating train pairs')
        train_pairs = dataset.generate_pairs(all_walks)
        logging.info('dumping train pairs to %s' % (train_pairs_file))
        pkl.dump(train_pairs, open(train_pairs_file, 'wb'))

    logging.info('total train pairs: %d' % (len(train_pairs)))
    data = dataset.fetch_batch(train_pairs,
                               config['data_loader']['args']['batch_size'])

    place = fluid.CUDAPlace(0) if config['use_cuda'] else fluid.CPUPlace()
    train_program = fluid.Program()
    startup_program = fluid.Program()
    test_program = fluid.Program()

    with fluid.program_guard(train_program, startup_program):
        model = getattr(Model, config['model']['type'])(
            config['model']['args'], dataset, place)

    test_program = train_program.clone(for_test=True)
    with fluid.program_guard(train_program, startup_program):
        global_steps = len(data) * config['trainer']['args']['epochs']
        model.backward(global_steps, config['optimizer']['args'])

    # train
    exe = fluid.Executor(place)
    exe.run(startup_program)
    feed_dict = model.gw.to_feed(dataset.graph)

    logging.info('test before training...')
    test(test_program, exe, dataset, model, feed_dict)
    logging.info('training...')
    for epoch in range(1, 1 + config['trainer']['args']['epochs']):
        train_result = run_epoch(epoch, config['trainer']['args'], dataset,
                                 data, train_program, test_program, model,
                                 feed_dict, exe)

        logging.info('validating and testing...')
        test_result = test(test_program, exe, dataset, model, feed_dict)

        filename = os.path.join(config['trainer']['args']['save_dir'],
                                'dict_embed_model_epoch_%d.pkl' % (epoch))
        save_model(test_program, exe, dataset, model, feed_dict, filename)

    logging.info(
        "final_test_AUC %.4f | final_test_PR %.4f | fianl_test_F1 %.4f" % (
            test_result['AUC'], test_result['PR'], test_result['F1']))

    logging.info('training finished')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GATNE')
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
