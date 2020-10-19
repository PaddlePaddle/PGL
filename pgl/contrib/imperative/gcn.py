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
import paddle.fluid as fluid

import pgl
from pgl import data_loader
from pgl.utils.logger import log
from pgl.contrib.imperative.graph_tensor import GraphTensor

import numpy as np
import time
import argparse

from pgl.math import segment_sum

# Proposal 0: Build in function. "sum, mean, max, min" 


# Proposal 1: User write segment_ids as args in the reduce_function。
def custom_reduce_sum(dst, message):
    return segment_sum(message, dst)


# Proposal 2: Pass segment_ids by the message object.
def custom_reduce_sum_message(message):
    return segment_sum(message.dst, message.msg)


# Proposal 3: Pass segment_ids by the Conv layers self.


class GCNConv(paddle.nn.Layer):
    def reduce_func(self, msg):
        out = self.linear_1(msg)

    def __init__(self, input_size, hidden_size, name):
        super(GCNConv, self).__init__()
        self.name = name

        self.hidden_size = hidden_size
        self.linear_1 = paddle.nn.Linear(
            input_size,
            hidden_size,
            weight_attr=fluid.ParamAttr(name=name + 'l1'))
        self.linear_2 = paddle.nn.Linear(
            hidden_size,
            hidden_size,
            weight_attr=fluid.ParamAttr(name=name + 'l2'))
        # dy create_parameter missing name
        self.bias = self.create_parameter(
            shape=[hidden_size], dtype='float32', is_bias=True)

    def send_src_copy(self, src_feat, dst_feat, edge_feat):
        return src_feat["h"]

    # Proposal 3: Pass segment_ids by the Conv layers self.
    def custom_reduce_sum(self, message):
        return segment_sum(message, self.graph.dst)

    def forward(self, gw, feature, norm=None, activation=None):
        self.graph = gw
        size = feature.shape[-1]
        if size > self.hidden_size:
            feature = self.linear_1(feature)
        if norm is not None:
            feature = feature * norm

        msg = gw.send(self.send_src_copy, nfeat_list=[("h", feature)])

        if size > self.hidden_size:
            # output = gw.recv(msg, "sum")
            # output = gw.recv(msg, custom_reduce_sum)
            output = gw.recv(msg, self.custom_reduce_sum)
        else:
            # output = gw.recv(msg, "sum")
            # output = gw.recv(msg, custom_reduce_sum)
            output = gw.recv(msg, self.custom_reduce_sum)
            output = self.linear_2(output)
        if norm is not None:
            output = output * norm

        output = paddle.add(output, self.bias)
        return output


class Cora(paddle.nn.Layer):
    def __init__(self, layers=2, input_size=16, hidden_size=16):
        super(Cora, self).__init__()
        self.conv_1 = GCNConv(input_size, hidden_size, name="gcn-layer-1")
        self.conv_2 = GCNConv(hidden_size, hidden_size, name="gcn-layer-2")

    def forward(self, pgraph, node_index, node_label):
        output = self.conv_1(pgraph, pgraph.node_feat['words'],
                             pgraph.node_feat['norm'])
        output = self.conv_2(pgraph, output, pgraph.node_feat['norm'])

        pred = paddle.gather(output, node_index)
        loss, pred = paddle.nn.functional.softmax_with_cross_entropy(
            logits=pred, label=node_label, return_softmax=True)
        acc = paddle.metric.accuracy(input=pred, label=node_label, k=1)
        loss = paddle.mean(loss)
        return loss, acc


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


def dymain(args):
    from paddle.optimizer import Adam
    from paddle import to_tensor

    paddle.disable_static()
    dataset = load(args.dataset)

    indegree = dataset.graph.indegree()
    norm = np.zeros_like(indegree, dtype="float32")
    norm[indegree > 0] = np.power(indegree[indegree > 0], -0.5)
    dataset.graph.node_feat["norm"] = np.expand_dims(norm, -1)

    hidden_size = 16

    pgraph = GraphTensor(dataset.graph)

    cora = Cora(input_size=1433)
    adam = Adam(learning_rate=1e-2, parameters=cora.parameters())

    train_index = dataset.train_index
    train_label = to_tensor(np.expand_dims(dataset.y[train_index], -1))
    train_index = to_tensor(np.expand_dims(train_index, -1))

    val_index = dataset.val_index
    val_label = to_tensor(np.expand_dims(dataset.y[val_index], -1))
    val_index = to_tensor(np.expand_dims(val_index, -1))

    test_index = dataset.test_index
    test_label = to_tensor(np.expand_dims(dataset.y[test_index], -1))
    test_index = to_tensor(np.expand_dims(test_index, -1))

    dur = []
    for epoch in range(2001):
        if epoch >= 10:
            t0 = time.time()
        train_loss, train_acc = cora(pgraph, train_index, train_label)
        train_loss.backward()
        adam.minimize(train_loss)
        cora.clear_gradients()
        if epoch >= 10:
            time_per_epoch = 1.0 * (time.time() - t0)
            dur.append(time_per_epoch)
        val_loss, val_acc = cora(pgraph, val_index, val_label)
        if epoch % 100 == 0:
            log.info("Epoch %d " % epoch + "(%.5lf sec) " % np.mean(
                dur) + "Train Loss: %f " % train_loss + "Train Acc: %f " %
                     train_acc + "Val Loss: %f " % val_loss + "Val Acc: %f " %
                     val_acc)

    test_loss, test_acc = cora(pgraph, test_index, test_label)
    log.info("Test acc {}".format(test_acc.numpy()))
    log.info("Training Over!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument(
        "--dataset", type=str, default="cora", help="dataset (cora, pubmed)")
    parser.add_argument("--use_cuda", action='store_true', help="use_cuda")
    args = parser.parse_args()
    log.info(args)
    dymain(args)
