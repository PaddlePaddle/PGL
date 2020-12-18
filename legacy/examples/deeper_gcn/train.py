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
#-*- coding: utf-8 -*-
import pgl
from pgl import data_loader
from pgl.utils.logger import log
import paddle.fluid as fluid
import numpy as np
import time
import argparse
from pgl.utils.log_writer import LogWriter # vdl
from model import DeeperGCN

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
    # vdl
    writer = LogWriter("checkpoints/train_history")

    dataset = load(args.dataset)
    place = fluid.CUDAPlace(0) if args.use_cuda else fluid.CPUPlace()
    train_program = fluid.Program()
    startup_program = fluid.Program()
    test_program = fluid.Program()
    hidden_size = 64
    num_layers = 7

    with fluid.program_guard(train_program, startup_program):
        gw = pgl.graph_wrapper.GraphWrapper(
            name="graph",
            node_feat=dataset.graph.node_feat_info())
        
        output = DeeperGCN(gw, 
                gw.node_feat["words"],
                num_layers,
                hidden_size,
                dataset.num_classes,
                "deepercnn",
                0.1)

        node_index = fluid.layers.data(
            "node_index",
            shape=[None, 1],
            dtype="int64",
            append_batch_size=False)
        node_label = fluid.layers.data(
            "node_label",
            shape=[None, 1],
            dtype="int64",
            append_batch_size=False)

        pred = fluid.layers.gather(output, node_index)
        loss, pred = fluid.layers.softmax_with_cross_entropy(
            logits=pred, label=node_label, return_softmax=True)
        acc = fluid.layers.accuracy(input=pred, label=node_label, k=1)
        loss = fluid.layers.mean(loss)

    test_program = train_program.clone(for_test=True)
    with fluid.program_guard(train_program, startup_program):
        adam = fluid.optimizer.Adam(
            regularization=fluid.regularizer.L2DecayRegularizer(
                regularization_coeff=0.0005),
            learning_rate=0.005)
        adam.minimize(loss)

    exe = fluid.Executor(place)
    exe.run(startup_program)

    feed_dict = gw.to_feed(dataset.graph)

    train_index = dataset.train_index
    train_label = np.expand_dims(dataset.y[train_index], -1)
    train_index = np.expand_dims(train_index, -1)

    val_index = dataset.val_index
    val_label = np.expand_dims(dataset.y[val_index], -1)
    val_index = np.expand_dims(val_index, -1)

    test_index = dataset.test_index
    test_label = np.expand_dims(dataset.y[test_index], -1)
    test_index = np.expand_dims(test_index, -1)
    
    # get beta param
    beta_param_list = []
    for param in fluid.io.get_program_parameter(train_program):
        if param.name.endswith("_beta"):
            beta_param_list.append(param)

    dur = []
    for epoch in range(200):
        if epoch >= 3:
            t0 = time.time()
        feed_dict["node_index"] = np.array(train_index, dtype="int64")
        feed_dict["node_label"] = np.array(train_label, dtype="int64")
        train_loss, train_acc = exe.run(train_program,
                                        feed=feed_dict,
                                        fetch_list=[loss, acc],
                                        return_numpy=True)
        for param in beta_param_list:
            beta = np.array(fluid.global_scope().find_var(param.name).get_tensor())
            writer.add_scalar("beta/"+param.name, beta, epoch)

        if epoch >= 3:
            time_per_epoch = 1.0 * (time.time() - t0)
            dur.append(time_per_epoch)

        feed_dict["node_index"] = np.array(val_index, dtype="int64")
        feed_dict["node_label"] = np.array(val_label, dtype="int64")
        val_loss, val_acc = exe.run(test_program,
                                    feed=feed_dict,
                                    fetch_list=[loss, acc],
                                    return_numpy=True)

        log.info("Epoch %d " % epoch + "(%.5lf sec) " % np.mean(dur) +
                 "Train Loss: %f " % train_loss + "Train Acc: %f " % train_acc
                 + "Val Loss: %f " % val_loss + "Val Acc: %f " % val_acc)

    feed_dict["node_index"] = np.array(test_index, dtype="int64")
    feed_dict["node_label"] = np.array(test_label, dtype="int64")
    test_loss, test_acc = exe.run(test_program,
                                  feed=feed_dict,
                                  fetch_list=[loss, acc],
                                  return_numpy=True)
    log.info("Accuracy: %f" % test_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeeperGCN')
    parser.add_argument(
        "--dataset", type=str, default="cora", help="dataset (cora, pubmed)")
    parser.add_argument("--use_cuda", action='store_true', help="use_cuda")
    args = parser.parse_args()
    log.info(args)
    main(args)
