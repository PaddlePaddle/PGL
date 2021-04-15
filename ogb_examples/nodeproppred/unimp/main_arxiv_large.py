# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import math
import torch
import paddle
import pgl
import numpy as np
import paddle.fluid as F
import paddle.fluid.layers as L
from pgl.contrib.ogb.nodeproppred.dataset_pgl import PglNodePropPredDataset
from ogb.nodeproppred import Evaluator
from utils import to_undirected, add_self_loop, linear_warmup_decay
from model_large import Arxiv_baseline_model, Arxiv_label_embedding_model
from optimization import optimization
import argparse
from tqdm import tqdm
evaluator = Evaluator(name='ogbn-arxiv')


def get_config():
    parser = argparse.ArgumentParser()

    ## model_base_arg
    model_group = parser.add_argument_group('model_base_arg')
    model_group.add_argument('--num_layers', default=3, type=int)
    model_group.add_argument('--hidden_size', default=80, type=int)
    model_group.add_argument('--num_heads', default=5, type=int)
    model_group.add_argument('--dropout', default=0.3, type=float)
    model_group.add_argument('--attn_dropout', default=0.1, type=float)

    ## embed_arg
    embed_group = parser.add_argument_group('embed_arg')
    embed_group.add_argument('--use_label_e', action='store_true')
    embed_group.add_argument('--label_rate', default=0.65, type=float)

    ## train_arg
    train_group = parser.add_argument_group('train_arg')
    train_group.add_argument('--runs', default=10, type=int)
    train_group.add_argument('--epochs', default=2000, type=int)
    train_group.add_argument('--lr', default=0.001, type=float)
    train_group.add_argument('--place', default=-1, type=int)
    train_group.add_argument(
        '--log_file', default='result_arxiv.txt', type=str)
    return parser.parse_args()


def optimizer_func(lr=0.01):
    return F.optimizer.AdamOptimizer(
        learning_rate=lr,
        regularization=F.regularizer.L2Decay(regularization_coeff=0.0005))


def eval_test(parser, program, model, test_exe, graph, y_true, split_idx):
    feed_dict = model.gw.to_feed(graph)
    if parser.use_label_e:
        feed_dict['label'] = y_true
        feed_dict['label_idx'] = split_idx['train']
        feed_dict['attn_drop'] = -1

    avg_cost_np = test_exe.run(program=program,
                               feed=feed_dict,
                               fetch_list=[model.out_feat])

    y_pred = avg_cost_np[0].argmax(axis=-1)
    y_pred = np.expand_dims(y_pred, 1)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    val_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, val_acc, test_acc


def train_loop(parser,
               start_program,
               main_program,
               test_program,
               model,
               graph,
               label,
               split_idx,
               exe,
               run_id,
               wf=None):

    exe.run(start_program)
    max_acc = 0
    max_step = 0
    max_val_acc = 0
    max_cor_acc = 0
    max_cor_step = 0

    for epoch_id in tqdm(range(parser.epochs)):

        if parser.use_label_e:
            feed_dict = model.gw.to_feed(graph)
            train_idx_temp = split_idx['train']
            np.random.shuffle(train_idx_temp)
            label_idx = train_idx_temp[:int(parser.label_rate * len(
                train_idx_temp))]
            unlabel_idx = train_idx_temp[int(parser.label_rate * len(
                train_idx_temp)):]
            feed_dict['label'] = label
            feed_dict['label_idx'] = label_idx
            feed_dict['train_idx'] = unlabel_idx
            feed_dict['attn_drop'] = parser.attn_dropout
        else:
            feed_dict = model.gw.to_feed(graph)
            feed_dict['label'] = label
            feed_dict['train_idx'] = split_idx['train']

        loss = exe.run(main_program,
                       feed=feed_dict,
                       fetch_list=[model.avg_cost])
        loss = loss[0]

        result = eval_test(parser, test_program, model, exe, graph, label,
                           split_idx)
        train_acc, valid_acc, test_acc = result

        max_val_acc = max(valid_acc, max_val_acc)
        if max_val_acc == valid_acc:
            max_cor_acc = test_acc
            max_cor_step = epoch_id

        if max_acc == result[2]:
            max_step = epoch_id
        result_t = (f'Run: {run_id:02d}, '
                    f'Epoch: {epoch_id:02d}, '
                    f'Loss: {loss[0]:.4f}, '
                    f'Train: {100 * train_acc:.2f}%, '
                    f'Valid: {100 * valid_acc:.2f}%, '
                    f'Test: {100 * test_acc:.2f}% \n'
                    f'max_val: {100 * max_val_acc:.2f}%, '
                    f'max_val_Test: {100 * max_cor_acc:.2f}%, '
                    f'max_val_step: {max_cor_step}\n')
        if (epoch_id + 1) % 100 == 0:
            print(result_t)
            wf.write(result_t)
            wf.write('\n')
            wf.flush()
    return max_cor_acc


if __name__ == '__main__':
    parser = get_config()
    print('===========args==============')
    print(parser)
    print('=============================')

    startup_prog = F.default_startup_program()
    train_prog = F.default_main_program()

    place = F.CPUPlace() if parser.place < 0 else F.CUDAPlace(parser.place)

    dataset = PglNodePropPredDataset(name="ogbn-arxiv")
    split_idx = dataset.get_idx_split()

    graph, label = dataset[0]
    print(label.shape)

    graph = to_undirected(graph)
    graph = add_self_loop(graph)

    with F.unique_name.guard():
        with F.program_guard(train_prog, startup_prog):
            gw = pgl.graph_wrapper.GraphWrapper(
                name="arxiv", node_feat=graph.node_feat_info(), place=place)

            if parser.use_label_e:
                model = Arxiv_label_embedding_model(
                    gw, parser.hidden_size, parser.num_heads, parser.dropout,
                    parser.num_layers)
            else:
                model = Arxiv_baseline_model(gw, parser.hidden_size,
                                             parser.num_heads, parser.dropout,
                                             parser.num_layers)

            test_prog = train_prog.clone(for_test=True)
            model.train_program()

            adam_optimizer = optimizer_func(parser.lr)
            adam_optimizer = F.optimizer.RecomputeOptimizer(adam_optimizer)
            adam_optimizer._set_checkpoints(model.checkpoints)
            adam_optimizer.minimize(model.avg_cost)

    exe = F.Executor(place)

    wf = open(parser.log_file, 'w', encoding='utf-8')
    total_test_acc = 0.0
    for run_i in range(parser.runs):
        total_test_acc += train_loop(parser, startup_prog, train_prog,
                                     test_prog, model, graph, label, split_idx,
                                     exe, run_i, wf)
    wf.write(f'average: {100 * (total_test_acc/parser.runs):.2f}%')
    wf.close()

# Runned 10 times
# Val Accs: [74.64, 74.74, 74.71, 74.83, 74.82, 74.77, 74.75, 74.86, 74.6, 74.76]
# Test Accs: [73.79, 73.82, 74.0, 73.85, 74.02, 73.67, 73.65, 73.87, 73.66, 73.6]
# Average val accuracy: 74.74799999999999 ± 0.0775628777186617
# Average test accuracy: 73.793 ± 0.13957435294494433
# params: 1162515
