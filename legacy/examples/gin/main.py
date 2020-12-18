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
This file implement the training process of GIN model.
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

from Dataset import GINDataset, fold10_split, random_split
from dataloader import GraphDataloader
from model import GINModel


def main(args):
    """main function"""
    dataset = GINDataset(
        args.data_path,
        args.dataset_name,
        self_loop=not args.train_eps,
        degree_as_nlabel=True)
    train_dataset, test_dataset = fold10_split(
        dataset, fold_idx=args.fold_idx, seed=args.seed)

    train_loader = GraphDataloader(train_dataset, batch_size=args.batch_size)
    test_loader = GraphDataloader(
        test_dataset, batch_size=args.batch_size, shuffle=False)

    place = fluid.CUDAPlace(0) if args.use_cuda else fluid.CPUPlace()
    train_program = fluid.Program()
    startup_program = fluid.Program()

    with fluid.program_guard(train_program, startup_program):
        gw = pgl.graph_wrapper.GraphWrapper(
            "gw", place=place, node_feat=dataset[0][0].node_feat_info())

        model = GINModel(args, gw, dataset.gclasses)
        model.forward()

    infer_program = train_program.clone(for_test=True)

    with fluid.program_guard(train_program, startup_program):
        epoch_step = int(len(train_dataset) / args.batch_size) + 1
        boundaries = [
            i
            for i in range(50 * epoch_step, args.epochs * epoch_step,
                           epoch_step * 50)
        ]
        values = [args.lr * 0.5**i for i in range(0, len(boundaries) + 1)]
        lr = fl.piecewise_decay(boundaries=boundaries, values=values)
        train_op = fluid.optimizer.Adam(lr).minimize(model.loss)

    exe = fluid.Executor(place)
    exe.run(startup_program)

    # train and evaluate
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        for idx, batch_data in enumerate(train_loader):
            g, labels = batch_data
            feed_dict = gw.to_feed(g)
            feed_dict['labels'] = labels
            ret_loss, ret_lr, ret_acc = exe.run(
                train_program,
                feed=feed_dict,
                fetch_list=[model.loss, lr, model.acc])

            global_step += 1
            if global_step % 10 == 0:
                message = "epoch %d | step %d | " % (epoch, global_step)
                message += "lr %.6f | loss %.6f | acc %.4f" % (
                    ret_lr, ret_loss, ret_acc)
                log.info(message)

        # evaluate
        result = evaluate(exe, infer_program, model, gw, test_loader)

        message = "evaluating result"
        for key, value in result.items():
            message += " | %s %.6f" % (key, value)
        log.info(message)


def evaluate(exe, prog, model, gw, loader):
    """evaluate"""
    total_loss = []
    total_acc = []
    for idx, batch_data in enumerate(loader):
        g, labels = batch_data
        feed_dict = gw.to_feed(g)
        feed_dict['labels'] = labels
        ret_loss, ret_acc = exe.run(prog,
                                    feed=feed_dict,
                                    fetch_list=[model.loss, model.acc])
        total_loss.append(ret_loss)
        total_acc.append(ret_acc)

    total_loss = np.mean(total_loss)
    total_acc = np.mean(total_acc)

    return {"loss": total_loss, "acc": total_acc}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./dataset')
    parser.add_argument('--dataset_name', type=str, default='MUTAG')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--fold_idx', type=int, default=0)
    parser.add_argument('--output_path', type=str, default='./outputs/')
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--num_mlp_layers', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument(
        '--pool_type',
        type=str,
        default="sum",
        choices=["sum", "average", "max"])
    parser.add_argument('--train_eps', action='store_true')
    parser.add_argument('--epochs', type=int, default=350)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dropout_prob', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    log.info(args)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    main(args)
