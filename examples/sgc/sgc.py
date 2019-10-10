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
This file implement the training process of SGC model with StaticGraphWrapper.
"""

import os
import argparse
import numpy as np
import random
import time

import pgl
from pgl import data_loader
from pgl.utils.logger import log
from pgl.utils import paddle_helper
import paddle.fluid as fluid


def load(name):
    """Load dataset."""
    if name == 'cora':
        dataset = data_loader.CoraDataset()
    elif name == "pubmed":
        dataset = data_loader.CitationDataset("pubmed", symmetry_edges=False)
    elif name == "citeseer":
        dataset = data_loader.CitationDataset("citeseer", symmetry_edges=False)
    else:
        raise ValueError(name + " dataset doesn't exists")
    return dataset


def expand_data_dim(dataset):
    """Expand the dimension of data."""
    train_index = dataset.train_index
    train_label = np.expand_dims(dataset.y[train_index], -1)
    train_index = np.expand_dims(train_index, -1)

    val_index = dataset.val_index
    val_label = np.expand_dims(dataset.y[val_index], -1)
    val_index = np.expand_dims(val_index, -1)

    test_index = dataset.test_index
    test_label = np.expand_dims(dataset.y[test_index], -1)
    test_index = np.expand_dims(test_index, -1)

    return {
        'train_index': train_index,
        'train_label': train_label,
        'val_index': val_index,
        'val_label': val_label,
        'test_index': test_index,
        'test_label': test_label,
    }


def MessagePassing(gw, feature, num_layers, norm=None):
    """Precomputing message passing.
    """

    def send_src_copy(src_feat, dst_feat, edge_feat):
        """send_src_copy
        """
        return src_feat["h"]

    for _ in range(num_layers):
        if norm is not None:
            feature = feature * norm

        msg = gw.send(send_src_copy, nfeat_list=[("h", feature)])

        feature = gw.recv(msg, "sum")

        if norm is not None:
            feature = feature * norm

    return feature


def pre_gather(features, name_prefix, node_index_val):
    """Get features with respect to node index.
    """
    node_index, init = paddle_helper.constant(
        "%s_node_index" % (name_prefix), dtype='int32', value=node_index_val)
    logits = fluid.layers.gather(features, node_index)

    return logits, init


def calculate_loss(name, np_cached_h, node_label_val, num_classes, args):
    """Calculate loss function.
    """
    initializer = []
    const_cached_h, init = paddle_helper.constant(
        "const_%s_cached_h" % name, dtype='float32', value=np_cached_h)
    initializer.append(init)

    node_label, init = paddle_helper.constant(
        "%s_node_label" % (name), dtype='int64', value=node_label_val)
    initializer.append(init)

    output = fluid.layers.fc(const_cached_h,
                             size=num_classes,
                             bias_attr=args.bias,
                             name='fc')

    loss, probs = fluid.layers.softmax_with_cross_entropy(
        logits=output, label=node_label, return_softmax=True)
    loss = fluid.layers.mean(loss)

    acc = None
    if name != 'train':
        acc = fluid.layers.accuracy(input=probs, label=node_label, k=1)

    return {
        'loss': loss,
        'acc': acc,
        'probs': probs,
        'initializer': initializer
    }


def main(args):
    """"Main function."""
    dataset = load(args.dataset)

    # normalize
    indegree = dataset.graph.indegree()
    norm = np.zeros_like(indegree, dtype="float32")
    norm[indegree > 0] = np.power(indegree[indegree > 0], -0.5)
    dataset.graph.node_feat["norm"] = np.expand_dims(norm, -1)

    data = expand_data_dim(dataset)

    place = fluid.CUDAPlace(0) if args.use_cuda else fluid.CPUPlace()
    precompute_program = fluid.Program()
    startup_program = fluid.Program()
    train_program = fluid.Program()
    val_program = train_program.clone(for_test=True)
    test_program = train_program.clone(for_test=True)

    # precompute message passing and gather
    initializer = []
    with fluid.program_guard(precompute_program, startup_program):
        gw = pgl.graph_wrapper.StaticGraphWrapper(
            name="graph", place=place, graph=dataset.graph)

        cached_h = MessagePassing(
            gw,
            gw.node_feat["words"],
            num_layers=args.num_layers,
            norm=gw.node_feat['norm'])

        train_cached_h, init = pre_gather(cached_h, 'train',
                                          data['train_index'])
        initializer.append(init)
        val_cached_h, init = pre_gather(cached_h, 'val', data['val_index'])
        initializer.append(init)
        test_cached_h, init = pre_gather(cached_h, 'test', data['test_index'])
        initializer.append(init)

    exe = fluid.Executor(place)
    gw.initialize(place)
    for init in initializer:
        init(place)

    # get train features, val features and test features 
    np_train_cached_h, np_val_cached_h, np_test_cached_h = exe.run(
        precompute_program,
        feed={},
        fetch_list=[train_cached_h, val_cached_h, test_cached_h],
        return_numpy=True)

    initializer = []
    with fluid.program_guard(train_program, startup_program):
        with fluid.unique_name.guard():
            train_handle = calculate_loss('train', np_train_cached_h,
                                          data['train_label'],
                                          dataset.num_classes, args)
            initializer += train_handle['initializer']
            adam = fluid.optimizer.Adam(
                learning_rate=args.lr,
                regularization=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=args.weight_decay))
            adam.minimize(train_handle['loss'])

    with fluid.program_guard(val_program, startup_program):
        with fluid.unique_name.guard():
            val_handle = calculate_loss('val', np_val_cached_h,
                                        data['val_label'], dataset.num_classes,
                                        args)
            initializer += val_handle['initializer']

    with fluid.program_guard(test_program, startup_program):
        with fluid.unique_name.guard():
            test_handle = calculate_loss('test', np_test_cached_h,
                                         data['test_label'],
                                         dataset.num_classes, args)
            initializer += test_handle['initializer']

    exe.run(startup_program)
    for init in initializer:
        init(place)

    dur = []
    for epoch in range(args.epochs):
        if epoch >= 3:
            t0 = time.time()
        train_loss_t = exe.run(train_program,
                               feed={},
                               fetch_list=[train_handle['loss']],
                               return_numpy=True)[0]

        if epoch >= 3:
            time_per_epoch = 1.0 * (time.time() - t0)
            dur.append(time_per_epoch)

        val_loss_t, val_acc_t = exe.run(
            val_program,
            feed={},
            fetch_list=[val_handle['loss'], val_handle['acc']],
            return_numpy=True)

        log.info("Epoch %d " % epoch + "(%.5lf sec) " % np.mean(
            dur) + "Train Loss: %f " % train_loss_t + "Val Loss: %f " %
                 val_loss_t + "Val Acc: %f " % val_acc_t)

    test_loss_t, test_acc_t = exe.run(
        test_program,
        feed={},
        fetch_list=[test_handle['loss'], test_handle['acc']],
        return_numpy=True)
    log.info("Test Accuracy: %f" % test_acc_t)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SGC')
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help="dataset (cora, pubmed, citeseer)")
    parser.add_argument("--use_cuda", action='store_true', help="use_cuda")
    parser.add_argument(
        "--seed", type=int, default=1667, help="global random seed")
    parser.add_argument("--lr", type=float, default=0.2, help="learning rate")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.000005,
        help="Weight for L2 loss")
    parser.add_argument(
        "--bias", action='store_true', default=False, help="flag to use bias")
    parser.add_argument(
        "--epochs", type=int, default=200, help="number of training epochs")
    parser.add_argument(
        "--num_layers", type=int, default=2, help="number of SGC layers")
    args = parser.parse_args()
    log.info(args)
    main(args)
