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
"""test ogb
"""
import argparse

import pgl
import numpy as np
import paddle.fluid as fluid
from pgl.contrib.ogb.linkproppred.dataset_pgl import PglLinkPropPredDataset
from pgl.utils import paddle_helper
from ogb.linkproppred import Evaluator


def send_func(src_feat, dst_feat, edge_feat):
    """send_func"""
    return src_feat["h"]


def recv_func(feat):
    """recv_func"""
    return fluid.layers.sequence_pool(feat, pool_type="sum")


class GNNModel(object):
    """GNNModel"""

    def __init__(self, name, num_nodes, emb_dim, num_layers):
        self.num_nodes = num_nodes
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.name = name

        self.src_nodes = fluid.layers.data(
            name='src_nodes',
            shape=[None, 1],
            dtype='int64', )

        self.dst_nodes = fluid.layers.data(
            name='dst_nodes',
            shape=[None, 1],
            dtype='int64', )

        self.edge_label = fluid.layers.data(
            name='edge_label',
            shape=[None, 1],
            dtype='float32', )

    def forward(self, graph):
        """forward"""
        h = fluid.layers.create_parameter(
            shape=[self.num_nodes, self.emb_dim],
            dtype="float32",
            name=self.name + "_embedding")
        #  edge_attr = fluid.layers.fc(graph.edge_feat["feat"], size=self.emb_dim)

        for layer in range(self.num_layers):
            msg = graph.send(
                send_func,
                nfeat_list=[("h", h)], )
            h = graph.recv(msg, recv_func)
            h = fluid.layers.fc(
                h,
                size=self.emb_dim,
                bias_attr=False,
                param_attr=fluid.ParamAttr(name=self.name + '_%s' % layer))
            h = h * graph.node_feat["norm"]
            bias = fluid.layers.create_parameter(
                shape=[self.emb_dim],
                dtype='float32',
                is_bias=True,
                name=self.name + '_bias_%s' % layer)
            h = fluid.layers.elementwise_add(h, bias, act="relu")

        src = fluid.layers.gather(h, self.src_nodes)
        dst = fluid.layers.gather(h, self.dst_nodes)
        edge_embed = src * dst
        pred = fluid.layers.fc(input=edge_embed,
                               size=1,
                               name=self.name + "_pred_output")

        prob = fluid.layers.sigmoid(pred)

        loss = fluid.layers.sigmoid_cross_entropy_with_logits(pred,
                                                              self.edge_label)
        loss = fluid.layers.reduce_mean(loss)

        return pred, prob, loss


def main():
    """main
    """
    # Training settings
    parser = argparse.ArgumentParser(description='Graph Dataset')
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='number of epochs to train (default: 100)')
    parser.add_argument(
        '--dataset',
        type=str,
        default="ogbl-ppa",
        help='dataset name (default: protein protein associations)')
    args = parser.parse_args()

    #place = fluid.CUDAPlace(0)
    place = fluid.CPUPlace()  # Dataset too big to use GPU

    ### automatic dataloading and splitting
    print("loadding dataset")
    dataset = PglLinkPropPredDataset(name=args.dataset)
    splitted_edge = dataset.get_edge_split()
    print(splitted_edge['train_edge'].shape)
    print(splitted_edge['train_edge_label'].shape)

    print("building evaluator")
    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    graph_data = dataset[0]
    print("num_nodes: %d" % graph_data.num_nodes)

    train_program = fluid.Program()
    startup_program = fluid.Program()
    test_program = fluid.Program()
    # degree normalize
    indegree = graph_data.indegree()
    norm = np.zeros_like(indegree, dtype="float32")
    norm[indegree > 0] = np.power(indegree[indegree > 0], -0.5)
    graph_data.node_feat["norm"] = np.expand_dims(norm, -1).astype("float32")

    with fluid.program_guard(train_program, startup_program):
        model = GNNModel(
            name="gnn",
            num_nodes=graph_data.num_nodes,
            emb_dim=64,
            num_layers=2)
        gw = pgl.graph_wrapper.GraphWrapper(
            "graph",
            place,
            node_feat=graph_data.node_feat_info(),
            edge_feat=graph_data.edge_feat_info())
        pred, prob, loss = model.forward(gw)

    val_program = train_program.clone(for_test=True)

    with fluid.program_guard(train_program, startup_program):
        adam = fluid.optimizer.Adam(
            learning_rate=1e-2,
            regularization=fluid.regularizer.L2DecayRegularizer(
                regularization_coeff=0.0005))
        adam.minimize(loss)

    exe = fluid.Executor(place)
    exe.run(startup_program)

    feed = gw.to_feed(graph_data)
    for epoch in range(1, args.epochs + 1):
        feed['src_nodes'] = splitted_edge["train_edge"][:, 0].reshape(-1, 1)
        feed['dst_nodes'] = splitted_edge["train_edge"][:, 1].reshape(-1, 1)
        feed['edge_label'] = splitted_edge["train_edge_label"].astype(
            "float32").reshape(-1, 1)
        res_loss, y_pred = exe.run(train_program,
                                   feed=feed,
                                   fetch_list=[loss, prob])
        print("Loss %s" % res_loss[0])

        result = {}
        print("Evaluating...")
        feed['src_nodes'] = splitted_edge["valid_edge"][:, 0].reshape(-1, 1)
        feed['dst_nodes'] = splitted_edge["valid_edge"][:, 1].reshape(-1, 1)
        feed['edge_label'] = splitted_edge["valid_edge_label"].astype(
            "float32").reshape(-1, 1)
        y_pred = exe.run(val_program, feed=feed, fetch_list=[prob])[0]
        input_dict = {
            "y_true": splitted_edge["valid_edge_label"],
            "y_pred": y_pred.reshape(-1, ),
        }
        result["valid"] = evaluator.eval(input_dict)

        feed['src_nodes'] = splitted_edge["test_edge"][:, 0].reshape(-1, 1)
        feed['dst_nodes'] = splitted_edge["test_edge"][:, 1].reshape(-1, 1)
        feed['edge_label'] = splitted_edge["test_edge_label"].astype(
            "float32").reshape(-1, 1)
        y_pred = exe.run(val_program, feed=feed, fetch_list=[prob])[0]
        input_dict = {
            "y_true": splitted_edge["test_edge_label"],
            "y_pred": y_pred.reshape(-1, ),
        }
        result["test"] = evaluator.eval(input_dict)
        print(result)


if __name__ == "__main__":
    main()
