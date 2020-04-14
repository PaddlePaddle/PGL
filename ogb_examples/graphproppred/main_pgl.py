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
from pgl.contrib.ogb.graphproppred.dataset_pgl import PglGraphPropPredDataset
from pgl.utils import paddle_helper
from ogb.graphproppred import Evaluator
from pgl.contrib.ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


def train(exe, batch_size, graph_wrapper, train_program, splitted_idx, dataset,
          evaluator, fetch_loss, fetch_pred):
    """Train"""
    graphs, labels = dataset[splitted_idx["train"]]
    perm = np.arange(0, len(graphs))
    np.random.shuffle(perm)
    start_batch = 0
    batch_no = 0
    pred_output = np.zeros_like(labels, dtype="float32")
    while start_batch < len(perm):
        batch_index = perm[start_batch:start_batch + batch_size]
        start_batch += batch_size
        batch_graph = pgl.graph.MultiGraph(graphs[batch_index])
        batch_label = labels[batch_index]
        batch_valid = (batch_label == batch_label).astype("float32")
        batch_label = np.nan_to_num(batch_label).astype("float32")
        feed_dict = graph_wrapper.to_feed(batch_graph)
        feed_dict["label"] = batch_label
        feed_dict["weight"] = batch_valid
        loss, pred = exe.run(train_program,
                             feed=feed_dict,
                             fetch_list=[fetch_loss, fetch_pred])
        pred_output[batch_index] = pred
        batch_no += 1
    print("train", evaluator.eval({"y_true": labels, "y_pred": pred_output}))


def evaluate(exe, batch_size, graph_wrapper, val_program, splitted_idx,
             dataset, mode, evaluator, fetch_pred):
    """Eval"""
    graphs, labels = dataset[splitted_idx[mode]]
    perm = np.arange(0, len(graphs))
    start_batch = 0
    batch_no = 0
    pred_output = np.zeros_like(labels, dtype="float32")
    while start_batch < len(perm):
        batch_index = perm[start_batch:start_batch + batch_size]
        start_batch += batch_size
        batch_graph = pgl.graph.MultiGraph(graphs[batch_index])
        feed_dict = graph_wrapper.to_feed(batch_graph)
        pred = exe.run(val_program, feed=feed_dict, fetch_list=[fetch_pred])
        pred_output[batch_index] = pred[0]
        batch_no += 1
    print(mode, evaluator.eval({"y_true": labels, "y_pred": pred_output}))


def send_func(src_feat, dst_feat, edge_feat):
    """Send"""
    return src_feat["h"] + edge_feat["h"]


class GNNModel(object):
    """GNNModel"""

    def __init__(self, name, emb_dim, num_task, num_layers):
        self.num_task = num_task
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.name = name
        self.atom_encoder = AtomEncoder(name=name, emb_dim=emb_dim)
        self.bond_encoder = BondEncoder(name=name, emb_dim=emb_dim)

    def forward(self, graph):
        """foward"""
        h_node = self.atom_encoder(graph.node_feat['feat'])
        h_edge = self.bond_encoder(graph.edge_feat['feat'])
        for layer in range(self.num_layers):
            msg = graph.send(
                send_func,
                nfeat_list=[("h", h_node)],
                efeat_list=[("h", h_edge)])
            h_node = graph.recv(msg, 'sum') + h_node
            h_node = fluid.layers.fc(h_node,
                                     size=self.emb_dim,
                                     name=self.name + '_%s' % layer,
                                     act="relu")
        graph_nodes = pgl.layers.graph_pooling(graph, h_node, "average")
        graph_pred = fluid.layers.fc(graph_nodes, self.num_task, name="final")
        return graph_pred


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
        default="ogbg-mol-tox21",
        help='dataset name (default: proteinfunc)')
    args = parser.parse_args()

    place = fluid.CPUPlace()  # Dataset too big to use GPU

    ### automatic dataloading and splitting
    dataset = PglGraphPropPredDataset(name=args.dataset)
    splitted_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    graph_data, label = dataset[:2]
    batch_graph = pgl.graph.MultiGraph(graph_data)
    graph_data = batch_graph

    train_program = fluid.Program()
    startup_program = fluid.Program()
    test_program = fluid.Program()
    # degree normalize
    graph_data.edge_feat["feat"] = graph_data.edge_feat["feat"].astype("int64")
    graph_data.node_feat["feat"] = graph_data.node_feat["feat"].astype("int64")

    model = GNNModel(
        name="gnn", num_task=dataset.num_tasks, emb_dim=64, num_layers=2)

    with fluid.program_guard(train_program, startup_program):
        gw = pgl.graph_wrapper.GraphWrapper(
            "graph",
            node_feat=graph_data.node_feat_info(),
            edge_feat=graph_data.edge_feat_info())
        pred = model.forward(gw)
        sigmoid_pred = fluid.layers.sigmoid(pred)

    val_program = train_program.clone(for_test=True)

    initializer = []
    with fluid.program_guard(train_program, startup_program):
        train_label = fluid.layers.data(
            name="label", dtype="float32", shape=[None, dataset.num_tasks])
        train_weight = fluid.layers.data(
            name="weight", dtype="float32", shape=[None, dataset.num_tasks])
        train_loss_t = fluid.layers.sigmoid_cross_entropy_with_logits(
            x=pred, label=train_label) * train_weight
        train_loss_t = fluid.layers.reduce_sum(train_loss_t)

        adam = fluid.optimizer.Adam(
            learning_rate=1e-2,
            regularization=fluid.regularizer.L2DecayRegularizer(
                regularization_coeff=0.0005))
        adam.minimize(train_loss_t)

    exe = fluid.Executor(place)
    exe.run(startup_program)

    for epoch in range(1, args.epochs + 1):
        print("Epoch", epoch)
        train(exe, 128, gw, train_program, splitted_idx, dataset, evaluator,
              train_loss_t, sigmoid_pred)
        evaluate(exe, 128, gw, val_program, splitted_idx, dataset, "valid",
                 evaluator, sigmoid_pred)
        evaluate(exe, 128, gw, val_program, splitted_idx, dataset, "test",
                 evaluator, sigmoid_pred)


if __name__ == "__main__":
    main()
