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
import pgl
from pgl import data_loader
from pgl.utils import paddle_helper
from pgl.utils.logger import log
import paddle.fluid as fluid
import numpy as np
import time

import argparse


def load(name):
    if name == 'cora':
        dataset = data_loader.CoraDataset()
    elif name == "pubmed":
        dataset = data_loader.CitationDataset("pubmed", symmetry_edges=False)
    elif name == "citeseer":
        dataset = data_loader.CitationDataset("citeseer", symmetry_edges=False)
    else:
        raise ValueError(name + " dataset doesn't exists")
    return dataset


def main(args):
    dataset = load(args.dataset)

    # normalize
    indegree = dataset.graph.indegree()
    norm = np.zeros_like(indegree, dtype="float32")
    norm[indegree > 0] = np.power(indegree[indegree > 0], -0.5)
    dataset.graph.node_feat["norm"] = np.expand_dims(norm, -1)

    train_index = dataset.train_index
    train_label = np.expand_dims(dataset.y[train_index], -1)
    train_index = np.expand_dims(train_index, -1)

    val_index = dataset.val_index
    val_label = np.expand_dims(dataset.y[val_index], -1)
    val_index = np.expand_dims(val_index, -1)

    test_index = dataset.test_index
    test_label = np.expand_dims(dataset.y[test_index], -1)
    test_index = np.expand_dims(test_index, -1)

    place = fluid.CUDAPlace(0) if args.use_cuda else fluid.CPUPlace()
    train_program = fluid.Program()
    startup_program = fluid.Program()
    test_program = fluid.Program()
    hidden_size = 16

    with fluid.program_guard(train_program, startup_program):
        gw = pgl.graph_wrapper.StaticGraphWrapper(
            name="graph", graph=dataset.graph, place=place)
        output = pgl.layers.gcn(gw,
                                gw.node_feat["words"],
                                hidden_size,
                                activation="relu",
                                norm=gw.node_feat['norm'],
                                name="gcn_layer_1")
        output = fluid.layers.dropout(
            output, 0.5, dropout_implementation='upscale_in_train')
        output = pgl.layers.gcn(gw,
                                output,
                                dataset.num_classes,
                                activation=None,
                                norm=gw.node_feat['norm'],
                                name="gcn_layer_2")

    val_program = train_program.clone(for_test=True)
    test_program = train_program.clone(for_test=True)

    initializer = []
    with fluid.program_guard(train_program, startup_program):
        train_node_index, init = paddle_helper.constant(
            "train_node_index", dtype="int64", value=train_index)
        initializer.append(init)

        train_node_label, init = paddle_helper.constant(
            "train_node_label", dtype="int64", value=train_label)
        initializer.append(init)
        pred = fluid.layers.gather(output, train_node_index)
        train_loss_t = fluid.layers.softmax_with_cross_entropy(
            logits=pred, label=train_node_label)
        train_loss_t = fluid.layers.reduce_mean(train_loss_t)

        adam = fluid.optimizer.Adam(
            learning_rate=1e-2,
            regularization=fluid.regularizer.L2DecayRegularizer(
                regularization_coeff=0.0005))
        adam.minimize(train_loss_t)

    with fluid.program_guard(val_program, startup_program):
        val_node_index, init = paddle_helper.constant(
            "val_node_index", dtype="int64", value=val_index)
        initializer.append(init)

        val_node_label, init = paddle_helper.constant(
            "val_node_label", dtype="int64", value=val_label)
        initializer.append(init)

        pred = fluid.layers.gather(output, val_node_index)
        val_loss_t, pred = fluid.layers.softmax_with_cross_entropy(
            logits=pred, label=val_node_label, return_softmax=True)
        val_acc_t = fluid.layers.accuracy(
            input=pred, label=val_node_label, k=1)
        val_loss_t = fluid.layers.reduce_mean(val_loss_t)

    with fluid.program_guard(test_program, startup_program):
        test_node_index, init = paddle_helper.constant(
            "test_node_index", dtype="int64", value=test_index)
        initializer.append(init)

        test_node_label, init = paddle_helper.constant(
            "test_node_label", dtype="int64", value=test_label)
        initializer.append(init)

        pred = fluid.layers.gather(output, test_node_index)
        test_loss_t, pred = fluid.layers.softmax_with_cross_entropy(
            logits=pred, label=test_node_label, return_softmax=True)
        test_acc_t = fluid.layers.accuracy(
            input=pred, label=test_node_label, k=1)
        test_loss_t = fluid.layers.reduce_mean(test_loss_t)

    exe = fluid.Executor(place)
    exe.run(startup_program)
    gw.initialize(place)
    for init in initializer:
        init(place)

    dur = []
    for epoch in range(200):
        if epoch >= 3:
            t0 = time.time()

        train_loss = exe.run(train_program,
                             feed={},
                             fetch_list=[train_loss_t],
                             return_numpy=True)
        train_loss = train_loss[0]

        if epoch >= 3:
            time_per_epoch = 1.0 * (time.time() - t0)
            dur.append(time_per_epoch)

        val_loss, val_acc = exe.run(val_program,
                                    feed={},
                                    fetch_list=[val_loss_t, val_acc_t],
                                    return_numpy=True)

        log.info("Epoch %d " % epoch + "(%.5lf sec) " % np.mean(
            dur) + "Train Loss: %f " % train_loss + "Val Loss: %f " % val_loss
                 + "Val Acc: %f " % val_acc)

    test_loss, test_acc = exe.run(test_program,
                                  feed={},
                                  fetch_list=[test_loss_t, test_acc_t],
                                  return_numpy=True)
    log.info("Accuracy: %f" % test_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument(
        "--dataset", type=str, default="cora", help="dataset (cora, pubmed)")
    parser.add_argument("--use_cuda", action='store_true', help="use_cuda")
    args = parser.parse_args()
    log.info(args)
    main(args)
