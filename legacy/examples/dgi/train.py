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
    Train
"""
import os
import pgl
from pgl import data_loader
from pgl.utils.logger import log
import paddle.fluid as fluid
import numpy as np
import time
import argparse


def load(name):
    """Load"""
    if name == 'cora':
        dataset = data_loader.CoraDataset()
    elif name == "pubmed":
        dataset = data_loader.CitationDataset("pubmed", symmetry_edges=False)
    elif name == "citeseer":
        dataset = data_loader.CitationDataset("citeseer", symmetry_edges=False)
    else:
        raise ValueError(name + " dataset doesn't exists")
    return dataset


def load_param(dirname, var_name_list):
    """load_param"""
    for var_name in var_name_list:
        var = fluid.global_scope().find_var(var_name)
        var_tensor = var.get_tensor()
        var_tmp = np.load(os.path.join(dirname, var_name + '.npy'))
        var_tensor.set(var_tmp, fluid.CPUPlace())


def main(args):
    """main"""
    dataset = load(args.dataset)

    # normalize
    indegree = dataset.graph.indegree()
    norm = np.zeros_like(indegree, dtype="float32")
    norm[indegree > 0] = np.power(indegree[indegree > 0], -0.5)
    dataset.graph.node_feat["norm"] = np.expand_dims(norm, -1)

    place = fluid.CUDAPlace(0) if args.use_cuda else fluid.CPUPlace()
    train_program = fluid.Program()
    startup_program = fluid.Program()
    test_program = fluid.Program()
    hidden_size = 512

    with fluid.program_guard(train_program, startup_program):
        gw = pgl.graph_wrapper.GraphWrapper(
            name="graph",
            node_feat=dataset.graph.node_feat_info())

        output = pgl.layers.gcn(gw,
                                gw.node_feat["words"],
                                hidden_size,
                                activation="relu",
                                norm=gw.node_feat['norm'],
                                name="gcn_layer_1")
        output.stop_gradient = True
        output = fluid.layers.fc(output,
                                 dataset.num_classes,
                                 act=None,
                                 name="classifier")
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
        adam = fluid.optimizer.Adam(learning_rate=1e-2)
        adam.minimize(loss)

    exe = fluid.Executor(place)
    exe.run(startup_program)

    load_param(args.checkpoint, ["gcn_layer_1", "gcn_layer_1_bias"])
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
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument(
        "--dataset", type=str, default="cora", help="dataset (cora, pubmed)")
    parser.add_argument(
        "--checkpoint", type=str, default="best_model", help="checkpoint")
    parser.add_argument("--use_cuda", action='store_true', help="use_cuda")
    args = parser.parse_args()
    log.info(args)
    main(args)
