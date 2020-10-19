# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
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
import paddle
import pgl
from pgl import data_loader
from pgl.utils.logger import log
import paddle.fluid as fluid
import numpy as np
import time
import argparse
import paddle.fluid as fluid
from pgl import graph_wrapper
from pgl.utils import paddle_helper

from pgl.utils import op
from pgl.utils import paddle_helper
import warnings
from pgl.math import segment_sum


def gcn(gw, feature, hidden_size, activation, name, norm=None):
    def send_src_copy(src_feat, dst_feat, edge_feat):
        return src_feat["h"]

    size = feature.shape[-1]
    if size > hidden_size:
        feature = fluid.layers.fc(feature,
                                  size=hidden_size,
                                  bias_attr=False,
                                  param_attr=fluid.ParamAttr(name=name))

    if norm is not None:
        feature = feature * norm

    msg = gw.send(send_src_copy, nfeat_list=[("h", feature)])

    if size > hidden_size:
        output = gw.recv(msg, "sum")
    else:
        output = gw.recv(msg, "sum")
        output = fluid.layers.fc(output,
                                 size=hidden_size,
                                 bias_attr=False,
                                 param_attr=fluid.ParamAttr(name=name))

    if norm is not None:
        output = output * norm

    bias = fluid.layers.create_parameter(
        shape=[hidden_size],
        dtype='float32',
        is_bias=True,
        name=name + '_bias')
    output = fluid.layers.elementwise_add(output, bias)
    return output


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
    paddle.enable_static()
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
    hidden_size = 16

    with fluid.program_guard(train_program, startup_program):
        gw = pgl.graph_wrapper.GraphWrapper(
            name="graph",
            place=place,
            node_feat=dataset.graph.node_feat_info())

        output = gcn(gw,
                     gw.node_feat["words"],
                     hidden_size,
                     activation="relu",
                     norm=gw.node_feat['norm'],
                     name="gcn_layer_1")
        output = fluid.layers.dropout(
            output, 0.5, dropout_implementation='upscale_in_train')
        output = gcn(gw,
                     output,
                     dataset.num_classes,
                     activation=None,
                     norm=gw.node_feat['norm'],
                     name="gcn_layer_2")
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
            learning_rate=1e-2,
            # regularization=fluid.regularizer.L2DecayRegularizer(
            #     regularization_coeff=0.0005)
        )
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

    dur = []
    for epoch in range(2001):
        if epoch >= 10:
            t0 = time.time()
        feed_dict["node_index"] = np.array(train_index, dtype="int64")
        feed_dict["node_label"] = np.array(train_label, dtype="int64")
        train_loss, train_acc = exe.run(train_program,
                                        feed=feed_dict,
                                        fetch_list=[loss, acc],
                                        return_numpy=True)

        if epoch >= 10:
            time_per_epoch = 1.0 * (time.time() - t0)
            dur.append(time_per_epoch)
        feed_dict["node_index"] = np.array(val_index, dtype="int64")
        feed_dict["node_label"] = np.array(val_label, dtype="int64")
        val_loss, val_acc = exe.run(test_program,
                                    feed=feed_dict,
                                    fetch_list=[loss, acc],
                                    return_numpy=True)
        if epoch % 100 == 0:
            log.info("Epoch %d " % epoch + "(%.5lf sec) " % np.mean(
                dur) + "Train Loss: %f " % train_loss + "Train Acc: %f " %
                     train_acc + "Val Loss: %f " % val_loss + "Val Acc: %f " %
                     val_acc)

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
    parser.add_argument("--use_cuda", action='store_true', help="use_cuda")
    args = parser.parse_args()
    log.info(args)
    main(args)
