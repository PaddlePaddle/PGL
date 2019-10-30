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
This file implement the training process of LINE model.
"""

import time
import argparse
import random
import os
import numpy as np

import pgl
import paddle.fluid as fluid
import paddle.fluid.layers as fl
from pgl.utils.logger import log

from data_loader import FlickrDataset


def make_dir(path):
    """Create directory if path is not existed.

    Args:
        path: The directory that wants to create.
    """
    try:
        os.makedirs(path)
    except:
        if not os.path.isdir(path):
            raise


def save_param(dirname, var_name_list):
    """save_param"""
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    for var_name in var_name_list:
        var = fluid.global_scope().find_var(var_name)
        var_tensor = var.get_tensor()
        np.save(os.path.join(dirname, var_name + '.npy'), np.array(var_tensor))


def set_seed(seed):
    """Set global random seed.
    """
    random.seed(seed)
    np.random.seed(seed)


def build_model(args, graph):
    """Build LINE model.

    Args:
        args: The hyperparameters for configure.
    
        graph: The :code:`Graph` data object.
        
    """
    u_i = fl.data(
        name='u_i', shape=[None, 1], dtype='int64', append_batch_size=False)
    u_j = fl.data(
        name='u_j', shape=[None, 1], dtype='int64', append_batch_size=False)

    label = fl.data(
        name='label', shape=[None], dtype='float32', append_batch_size=False)

    lr = fl.data(
        name='learning_rate',
        shape=[1],
        dtype='float32',
        append_batch_size=False)

    u_i_embed = fl.embedding(
        input=u_i,
        size=[graph.num_nodes, args.embed_dim],
        param_attr='shared_w')

    if args.order == 'first_order':
        u_j_embed = fl.embedding(
            input=u_j,
            size=[graph.num_nodes, args.embed_dim],
            param_attr='shared_w')
    elif args.order == 'second_order':
        u_j_embed = fl.embedding(
            input=u_j,
            size=[graph.num_nodes, args.embed_dim],
            param_attr='context_w')
    else:
        raise ValueError("order should be first_order or second_order, not %s"
                         % (args.order))

    inner_product = fl.reduce_sum(u_i_embed * u_j_embed, dim=1)

    loss = -1 * fl.reduce_mean(fl.logsigmoid(label * inner_product))
    optimizer = fluid.optimizer.RMSPropOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(loss)

    return loss, optimizer


def main(args):
    """The main funciton for training LINE model.
    """
    make_dir(args.save_dir)
    set_seed(args.seed)

    dataset = FlickrDataset(args.data_path)

    log.info('num nodes in graph: %d' % dataset.graph.num_nodes)
    log.info('num edges in graph: %d' % dataset.graph.num_edges)

    place = fluid.CUDAPlace(0) if args.use_cuda else fluid.CPUPlace()

    main_program = fluid.default_main_program()
    startup_program = fluid.default_startup_program()

    # build model here
    with fluid.program_guard(main_program, startup_program):
        loss, opt = build_model(args, dataset.graph)

    exe = fluid.Executor(place)
    exe.run(startup_program)  #initialize the parameters of the network

    batchrange = int(dataset.graph.num_edges / args.batch_size)
    T = batchrange * args.epochs
    for epoch in range(args.epochs):
        for b in range(batchrange):
            lr = max(args.lr * (1 - (batchrange * epoch + b) / T), 0.0001)

            u_i, u_j, label = dataset.fetch_batch(
                batch_size=args.batch_size,
                K=args.neg_sample_size,
                edge_sampling=args.sample_method,
                node_sampling=args.sample_method)

            feed_dict = {
                'u_i': u_i,
                'u_j': u_j,
                'label': label,
                'learning_rate': lr
            }

            ret_loss = exe.run(main_program,
                               feed=feed_dict,
                               fetch_list=[loss],
                               return_numpy=True)

            if b % 500 == 0:
                log.info("Epoch %d | Step %d | Loss %f | lr: %f" %
                         (epoch, b, ret_loss[0], lr))

        # save parameters in every epoch
        log.info("saving persistables parameters...")
        cur_save_path = os.path.join(args.save_dir,
                                     "model_epoch_%d" % (epoch + 1))
        save_param(cur_save_path, ['shared_w'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LINE')
    parser.add_argument(
        '--data_path',
        type=str,
        default='./data/flickr/',
        help='dataset for training')
    parser.add_argument("--use_cuda", action='store_true', help="use_cuda")
    parser.add_argument("--epochs", type=int, default=20, help='total epochs')
    parser.add_argument("--seed", type=int, default=1667, help='random seed')
    parser.add_argument("--lr", type=float, default=0.01, help='learning rate')
    parser.add_argument(
        "--neg_sample_size",
        type=int,
        default=5,
        help='negative samplle number')
    parser.add_argument("--save_dir", type=str, default="./checkpoints/model")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=128,
        help='the dimension of node embedding')
    parser.add_argument(
        "--sample_method",
        type=str,
        default="alias",
        help='negative sample method (uniform, numpy, alias)')
    parser.add_argument(
        "--order",
        type=str,
        default="first_order",
        help='the order of neighbors (first_order, second_order)')

    args = parser.parse_args()

    main(args)
