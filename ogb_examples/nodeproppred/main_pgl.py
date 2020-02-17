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
from pgl.contrib.ogb.nodeproppred.dataset_pgl import PglNodePropPredDataset
from pgl.utils import paddle_helper
from ogb.nodeproppred import Evaluator


def train():
    pass


def send_func(src_feat, dst_feat, edge_feat):
    return (src_feat["h"] + edge_feat["h"]) * src_feat["norm"]


class GNNModel(object):
    def __init__(self, name, emb_dim, num_task, num_layers):
        self.num_task = num_task
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.name = name

    def forward(self, graph):
        h = fluid.layers.embedding(
            graph.node_feat["x"],
            size=(2, self.emb_dim))  # name=self.name + "_embedding") 
        edge_attr = fluid.layers.fc(graph.edge_feat["feat"], size=self.emb_dim)
        for layer in range(self.num_layers):
            msg = graph.send(
                send_func,
                nfeat_list=[("h", h), ("norm", graph.node_feat["norm"])],
                efeat_list=[("h", edge_attr)])
            h = graph.recv(msg, "sum")
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
        pred = fluid.layers.fc(h,
                               self.num_task,
                               act=None,
                               name=self.name + "_pred_output")
        return pred


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
        default="ogbn-proteins",
        help='dataset name (default: proteinfunc)')
    args = parser.parse_args()

    #device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    #place = fluid.CUDAPlace(0)
    place = fluid.CPUPlace()  # Dataset too big to use GPU

    ### automatic dataloading and splitting
    dataset = PglNodePropPredDataset(name=args.dataset)
    splitted_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    graph_data, label = dataset[0]

    train_program = fluid.Program()
    startup_program = fluid.Program()
    test_program = fluid.Program()
    # degree normalize
    indegree = graph_data.indegree()
    norm = np.zeros_like(indegree, dtype="float32")
    norm[indegree > 0] = np.power(indegree[indegree > 0], -0.5)
    graph_data.node_feat["norm"] = np.expand_dims(norm, -1).astype("float32")
    graph_data.node_feat["x"] = np.zeros((len(indegree), 1), dtype="int64")
    graph_data.edge_feat["feat"] = graph_data.edge_feat["feat"].astype(
        "float32")
    model = GNNModel(
        name="gnn", num_task=dataset.num_tasks, emb_dim=64, num_layers=2)

    with fluid.program_guard(train_program, startup_program):
        gw = pgl.graph_wrapper.StaticGraphWrapper("graph", graph_data, place)
        pred = model.forward(gw)
        sigmoid_pred = fluid.layers.sigmoid(pred)

    val_program = train_program.clone(for_test=True)

    initializer = []
    with fluid.program_guard(train_program, startup_program):
        train_node_index, init = paddle_helper.constant(
            "train_node_index", dtype="int64", value=splitted_idx["train"])
        initializer.append(init)

        train_node_label, init = paddle_helper.constant(
            "train_node_label",
            dtype="float32",
            value=label[splitted_idx["train"]].astype("float32"))
        initializer.append(init)
        train_pred_t = fluid.layers.gather(pred, train_node_index)
        train_loss_t = fluid.layers.sigmoid_cross_entropy_with_logits(
            x=train_pred_t, label=train_node_label)
        train_loss_t = fluid.layers.reduce_sum(train_loss_t)
        train_pred_t = fluid.layers.sigmoid(train_pred_t)

        adam = fluid.optimizer.Adam(
            learning_rate=1e-2,
            regularization=fluid.regularizer.L2DecayRegularizer(
                regularization_coeff=0.0005))
        adam.minimize(train_loss_t)

    exe = fluid.Executor(place)
    exe.run(startup_program)
    gw.initialize(place)
    for init in initializer:
        init(place)

    for epoch in range(1, args.epochs + 1):
        loss = exe.run(train_program, feed={}, fetch_list=[train_loss_t])
        print("Loss %s" % loss[0])
        print("Evaluating...")
        y_pred = exe.run(val_program, feed={}, fetch_list=[sigmoid_pred])[0]
        result = {}
        input_dict = {
            "y_true": label[splitted_idx["train"]],
            "y_pred": y_pred[splitted_idx["train"]]
        }
        result["train"] = evaluator.eval(input_dict)
        input_dict = {
            "y_true": label[splitted_idx["valid"]],
            "y_pred": y_pred[splitted_idx["valid"]]
        }
        result["valid"] = evaluator.eval(input_dict)
        input_dict = {
            "y_true": label[splitted_idx["test"]],
            "y_pred": y_pred[splitted_idx["test"]]
        }
        result["test"] = evaluator.eval(input_dict)
        print(result)


if __name__ == "__main__":
    main()
