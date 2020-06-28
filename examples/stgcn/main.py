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
"""
This file implement the training process of STGCN model.
"""

import os
import sys
import time
import argparse
import numpy as np

import paddle.fluid as fluid
import paddle.fluid.layers as fl
import pgl
from pgl.utils.logger import log

from data_loader.data_utils import data_gen_mydata, gen_batch
from data_loader.graph import GraphFactory
from models.model import STGCNModel
from models.tester import model_inference, model_test


def main(args):
    """main"""
    PeMS = data_gen_mydata(args.input_file, args.label_file, args.n_route,
                           args.n_his, args.n_pred, (args.n_val, args.n_test))

    log.info(PeMS.get_stats())
    log.info(PeMS.get_len('train'))

    gf = GraphFactory(args)

    place = fluid.CUDAPlace(0) if args.use_cuda else fluid.CPUPlace()
    train_program = fluid.Program()
    startup_program = fluid.Program()

    with fluid.program_guard(train_program, startup_program):
        gw = pgl.graph_wrapper.GraphWrapper(
            "gw",
            node_feat=[('norm', [None, 1], "float32")],
            edge_feat=[('weights', [None, 1], "float32")])

        model = STGCNModel(args, gw)
        train_loss, y_pred = model.forward()

    infer_program = train_program.clone(for_test=True)

    with fluid.program_guard(train_program, startup_program):
        epoch_step = int(PeMS.get_len('train') / args.batch_size) + 1
        lr = fl.exponential_decay(
            learning_rate=args.lr,
            decay_steps=5 * epoch_step,
            decay_rate=0.7,
            staircase=True)
        if args.opt == 'RMSProp':
            train_op = fluid.optimizer.RMSPropOptimizer(lr).minimize(
                train_loss)
        elif args.opt == 'ADAM':
            train_op = fluid.optimizer.Adam(lr).minimize(train_loss)

    exe = fluid.Executor(place)
    exe.run(startup_program)

    if args.inf_mode == 'sep':
        # for inference mode 'sep', the type of step index is int.
        step_idx = args.n_pred - 1
        tmp_idx = [step_idx]
        min_val = min_va_val = np.array([4e1, 1e5, 1e5])
    elif args.inf_mode == 'merge':
        # for inference mode 'merge', the type of step index is np.ndarray.
        step_idx = tmp_idx = np.arange(3, args.n_pred + 1, 3) - 1
        min_val = min_va_val = np.array([4e1, 1e5, 1e5]) * len(step_idx)
    else:
        raise ValueError(f'ERROR: test mode "{args.inf_mode}" is not defined.')

    step = 0
    for epoch in range(1, args.epochs + 1):
        for idx, x_batch in enumerate(
                gen_batch(
                    PeMS.get_data('train'),
                    args.batch_size,
                    dynamic_batch=True,
                    shuffle=True)):

            x = np.array(x_batch[:, 0:args.n_his, :, :], dtype=np.float32)
            graph = gf.build_graph(x)
            feed = gw.to_feed(graph)
            feed['input'] = np.array(
                x_batch[:, 0:args.n_his + 1, :, :], dtype=np.float32)
            b_loss, b_lr = exe.run(train_program,
                                   feed=feed,
                                   fetch_list=[train_loss, lr])

            if idx % 5 == 0:
                log.info("epoch %d | step %d | lr %.6f | loss %.6f" %
                         (epoch, idx, b_lr[0], b_loss[0]))

        min_va_val, min_val = \
                model_inference(exe, gw, gf, infer_program, y_pred, PeMS, args, \
                                step_idx, min_va_val, min_val)

        for ix in tmp_idx:
            va, te = min_va_val[ix - 2:ix + 1], min_val[ix - 2:ix + 1]
            print(f'Time Step {ix + 1}: '
                  f'MAPE {va[0]:7.3%}, {te[0]:7.3%}; '
                  f'MAE  {va[1]:4.3f}, {te[1]:4.3f}; '
                  f'RMSE {va[2]:6.3f}, {te[2]:6.3f}.')

        if epoch % 5 == 0:
            model_test(exe, gw, gf, infer_program, y_pred, PeMS, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_route', type=int, default=5)
    parser.add_argument('--n_his', type=int, default=23)
    parser.add_argument('--n_pred', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save', type=int, default=10)
    parser.add_argument('--Ks', type=int, default=3)  #equal to num_layers
    parser.add_argument('--Kt', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--keep_prob', type=float, default=1.0)
    parser.add_argument('--opt', type=str, default='RMSProp')
    parser.add_argument('--inf_mode', type=str, default='sep')
    parser.add_argument('--input_file', type=str, default='dataset/input.csv')
    parser.add_argument('--label_file', type=str, default='dataset/output.csv')
    parser.add_argument(
        '--city_file', type=str, default='dataset/crawl_list.csv')
    parser.add_argument('--adj_mat_file', type=str, default='dataset/W_74.csv')
    parser.add_argument('--output_path', type=str, default='./outputs/')
    parser.add_argument('--n_val', type=str, default=1)
    parser.add_argument('--n_test', type=str, default=1)
    parser.add_argument('--use_cuda', action='store_true')
    args = parser.parse_args()

    blocks = [[1, 32, 64], [64, 32, 128]]
    args.blocks = blocks
    log.info(args)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    main(args)
