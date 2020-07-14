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

import paddle
import pgl
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.contrib import summary


def rgcn_conv(graph_wrapper,
              feature,
              hidden_size,
              edge_types,
              name="rgcn_conv"):
    def __message(src_feat, dst_feat, edge_feat):
        """send function
        """
        return src_feat['h']

    def __reduce(feat):
        """recv function
        """
        return fluid.layers.sequence_pool(feat, pool_type='average')

    if not isinstance(edge_types, list):
        edge_types = [edge_types]

    output = None
    for i in range(len(edge_types)):
        tmp_feat = fluid.layers.fc(
            feature,
            size=hidden_size,
            param_attr=fluid.ParamAttr(name='%s_node_fc_%s' %
                                       (name, edge_types[i].split("2")[0])),
            act=None)
        if output is None:
            output = fluid.layers.zeros_like(tmp_feat)
        msg = graph_wrapper[edge_types[i]].send(
            __message, nfeat_list=[('h', tmp_feat)])
        neigh_feat = graph_wrapper[edge_types[i]].recv(msg, __reduce)
        # The weight of FC should be the same for the same type of node
        # The edge type str should be `A2B`(from type A to type B)
        neigh_feat = fluid.layers.fc(
            neigh_feat,
            size=hidden_size,
            param_attr=fluid.ParamAttr(name='%s_edge_fc_%s' %
                                       (name, edge_types[i])),
            act=None)
        # TODO: the tmp_feat and neigh_feat should be add togather.
        output = output + neigh_feat * tmp_feat

    return output


class RGCNModel:
    def __init__(self, graph_wrapper, num_layers, hidden_size, num_class,
                 edge_types):
        self.graph_wrapper = graph_wrapper
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_class = num_class
        self.edge_types = edge_types

    def forward(self, feat):
        for i in range(self.num_layers - 1):
            feat = rgcn_conv(
                self.graph_wrapper,
                feat,
                self.hidden_size,
                self.edge_types,
                name="rgcn_%d" % i)
            feat = fluid.layers.relu(feat)
            feat = fluid.layers.dropout(feat, dropout_prob=0.5)
        feat = rgcn_conv(
            self.graph_wrapper,
            feat,
            self.num_class,
            self.edge_types,
            name="rgcn_%d" % (self.num_layers - 1))
        return feat


def cross_entropy_loss(logit, label):
    loss = fluid.layers.softmax_with_cross_entropy(logit, label)
    loss = fluid.layers.mean(loss)
    acc = fluid.layers.accuracy(fluid.layers.softmax(logit), label)
    return loss, acc


if __name__ == "__main__":
    num_nodes = 4
    num_class = 2
    feat_dim = 16
    hidden_size = 16
    num_layers = 2

    node_types = [(0, 'user'), (1, 'user'), (2, 'item'), (3, 'item')]
    edges = {
        'U2U': [(0, 1), (1, 0)],
        'U2I': [(1, 2), (0, 3), (1, 3)],
        'I2I': [(2, 3), (3, 2)],
    }
    node_feat = {'feature': np.random.randn(4, feat_dim)}
    edges_feat = {
        'U2U': {
            'h': np.random.randn(2, feat_dim)
        },
        'U2I': {
            'h': np.random.randn(3, feat_dim)
        },
        'I2I': {
            'h': np.random.randn(2, feat_dim)
        },
    }
    g = pgl.heter_graph.HeterGraph(
        num_nodes=num_nodes,
        edges=edges,
        node_types=node_types,
        node_feat=node_feat,
        edge_feat=edges_feat)

    train_program = fluid.Program()
    startup_program = fluid.Program()
    test_program = fluid.Program()

    with fluid.program_guard(train_program, startup_program):
        label = fluid.layers.data(shape=[-1], dtype="int64", name='label')
        label = fluid.layers.reshape(label, [-1, 1])
        label.stop_gradient = True
        gw = pgl.heter_graph_wrapper.HeterGraphWrapper(
            name="heter_graph",
            edge_types=g.edge_types_info(),
            node_feat=g.node_feat_info(),
            edge_feat=g.edge_feat_info())

        feat = fluid.layers.create_parameter(
            shape=[num_nodes, feat_dim], dtype='float32')

        model = RGCNModel(gw, num_layers, num_class, hidden_size,
                          g.edge_types_info())
        logit = model.forward(feat)
        loss, acc = cross_entropy_loss(logit, label)
        opt = fluid.optimizer.SGD(learning_rate=0.1)
        opt.minimize(loss)

    summary(train_program)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_program)

    feed_dict = gw.to_feed(g)
    feed_dict['label'] = np.array([1, 0, 1, 1]).astype('int64')
    for i in range(100):
        res = exe.run(train_program,
                      feed=feed_dict,
                      fetch_list=[loss.name, acc.name])
        print("STEP: %d  LOSS: %f  ACC: %f" % (i, res[0], res[1]))
