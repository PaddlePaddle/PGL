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
import paddle.fluid as fluid
import numpy as np
from pgl.contrib.ogb.nodeproppred.dataset_pgl import PglNodePropPredDataset


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

    gw = graph_wrapper
    if not isinstance(edge_types, list):
        edge_types = [edge_types]

    #output = fluid.layers.zeros((feature.shape[0], hidden_size), dtype='float32')
    output = None
    for i in range(len(edge_types)):
        assert feature is not None
        tmp_feat = fluid.layers.fc(
            feature,
            size=hidden_size,
            param_attr=fluid.ParamAttr(name='%s_node_fc_%s' %
                                       (name, edge_types[i].split("2")[0])),
            act=None)
        if output is None:
            output = fluid.layers.zeros_like(tmp_feat)
        msg = gw[edge_types[i]].send(__message, nfeat_list=[('h', feature)])
        neigh_feat = gw[edge_types[i]].recv(msg, __reduce)
        # The weight of FC should be the same for the same type of node
        # The edge type str should be `A2B`(from type A to type B)
        neigh_feat = fluid.layers.fc(
            neigh_feat,
            size=hidden_size,
            param_attr=fluid.ParamAttr(name='%s_edge_fc_%s' %
                                       (name, edge_types[i])),
            act=None)
        output = output + neigh_feat * tmp_feat
    #output = fluid.layers.relu(out)
    return output


class RGCNModel:
    def __init__(self, gw, layers, num_class, num_nodes, edge_types):
        self.hidden_size = 64
        self.layers = layers
        self.num_nodes = num_nodes
        self.edge_types = edge_types
        self.gw = gw
        self.num_class = num_class

    def forward(self, feat):
        for i in range(self.layers - 1):
            feat = rgcn_conv(
                self.gw,
                feat,
                self.hidden_size,
                self.edge_types,
                name="rgcn_%d" % i)
            feat = fluid.layers.relu(feat)
            feat = fluid.layers.dropout(feat, dropout_prob=0.5)
        feat = rgcn_conv(
            self.gw,
            feat,
            self.num_class,
            self.edge_types,
            name="rgcn_%d" % (self.layers - 1))
        return feat


def softmax_loss(feat, label, class_num):
    #logit = fluid.layers.fc(feat, class_num)
    logit = feat
    loss = fluid.layers.softmax_with_cross_entropy(logit, label)
    loss = fluid.layers.mean(loss)
    acc = fluid.layers.accuracy(fluid.layers.softmax(logit), label)
    return loss, logit, acc


def paper_mask(feat, gw, start_index):
    mask = fluid.layers.cast(gw[0].node_feat['index'] > start_index)
    feat = fluid.layers.mask_select(feat, mask)
    return feat


if __name__ == "__main__":
    #PglNodePropPredDataset('ogbn-mag')
    num_nodes = 4
    num_class = 2
    node_types = [(0, 'user'), (1, 'user'), (2, 'item'), (3, 'item')]
    edges = {
        'U2U': [(0, 1), (1, 0)],
        'U2I': [(1, 2), (0, 3), (1, 3)],
        'I2I': [(2, 3), (3, 2)],
    }
    node_feat = {'feature': np.random.randn(4, 16)}
    edges_feat = {
        'U2U': {
            'h': np.random.randn(2, 16)
        },
        'U2I': {
            'h': np.random.randn(3, 16)
        },
        'I2I': {
            'h': np.random.randn(2, 16)
        },
    }
    g = pgl.heter_graph.HeterGraph(
        num_nodes=num_nodes,
        edges=edges,
        node_types=node_types,
        node_feat=node_feat,
        edge_feat=edges_feat)
    place = fluid.CPUPlace()
    train_program = fluid.Program()
    startup_program = fluid.Program()
    test_program = fluid.Program()

    with fluid.program_guard(train_program, startup_program):
        label = fluid.layers.data(shape=[-1], dtype="int64", name='label')
        #label = fluid.layers.create_global_var(shape=[4], value=1, dtype="int64")
        label = fluid.layers.reshape(label, [-1, 1])
        label.stop_gradient = True
        gw = pgl.heter_graph_wrapper.HeterGraphWrapper(
            name="heter_graph",
            edge_types=g.edge_types_info(),
            node_feat=g.node_feat_info(),
            edge_feat=g.edge_feat_info())

        feat = fluid.layers.create_parameter(
            shape=[num_nodes, 16], dtype='float32')

        model = RGCNModel(gw, 3, num_class, num_nodes, g.edge_types_info())
        feat = model.forward(feat)
        loss, logit, acc = softmax_loss(feat, label, 2)
        opt = fluid.optimizer.AdamOptimizer(learning_rate=0.001)
        opt.minimize(loss)
    from paddle.fluid.contrib import summary
    summary(train_program)

    exe = fluid.Executor(place)
    exe.run(startup_program)
    feed_dict = gw.to_feed(g)
    feed_dict['label'] = np.array([1, 0, 1, 1]).astype('int64')
    for i in range(100):
        res = exe.run(train_program,
                      feed=feed_dict,
                      fetch_list=[loss.name, logit.name, acc.name])
        print("%d %f %f" % (i, res[0], res[2]))
        #print(res[1])
