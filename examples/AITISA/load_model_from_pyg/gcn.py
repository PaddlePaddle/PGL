# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import argparse
import numpy as np


class PyGGCN(torch.nn.Module):
    def __init__(self, input_size, num_class=1, hidden_size=64):
        super(PyGGCN, self).__init__()
        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, num_class)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edges):
        x, edge_index = x, edges.T
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class PGLGCN(paddle.nn.Layer):
    def __init__(self, input_size, num_class, hidden_size=64):
        super(PGLGCN, self).__init__()
        self.conv1 = pgl.nn.GCNConv(input_size, hidden_size)
        self.conv2 = pgl.nn.GCNConv(hidden_size, num_class)

    def forward(self, x, edges):
        x, edge_index = x, edges
        g = pgl.Graph(num_nodes=x.shape[0], edges=edges)
        x = paddle.nn.functional.relu(self.conv1(g, x))
        x = self.conv2(g, x)
        return x


def load(name):
    if name == 'cora':
        dataset = pgl.dataset.CoraDataset()
    elif name == "pubmed":
        dataset = pgl.dataset.CitationDataset("pubmed", symmetry_edges=True)
    elif name == "citeseer":
        dataset = pgl.dataset.CitationDataset("citeseer", symmetry_edges=True)
    else:
        raise ValueError(name + " dataset doesn't exists")

    return dataset


def convert_pyg2pgl(pgl_model, pyg_model):

    pyg_state_dict = pyg_model.state_dict()
    mapping = {
        "conv1.bias": "conv1.bias",
        "conv2.bias": "conv2.bias",
        "conv1.linear.weight": "conv1.weight",
        "conv2.linear.weight": "conv2.weight"
    }

    for key, value in pgl_model.state_dict().items():
        if mapping[key] in pyg_state_dict:
            print("Load key", key, " from PyG")
            value.set_value(pyg_state_dict[mapping[key]].cpu().numpy())
        else:
            print("Not found key", key, " from PyG")


def main(args):
    dataset = load(args.dataset)
    graph = dataset.graph
    x = graph.node_feat["words"]
    edges = graph.edges

    pgl_gnn_model = PGLGCN(
        input_size=x.shape[1], num_class=dataset.num_classes, hidden_size=16)

    pyg_gnn_model = PyGGCN(
        input_size=x.shape[1], num_class=dataset.num_classes, hidden_size=16)

    # Define Model in PGL
    convert_pyg2pgl(pgl_gnn_model, pyg_gnn_model)

    pgl_out = pgl_gnn_model(paddle.to_tensor(x), paddle.to_tensor(edges))
    pyg_out = pyg_gnn_model(torch.FloatTensor(x), torch.LongTensor(edges))

    eps = np.mean((pgl_out.numpy() - pyg_out.detach().numpy())**2)
    print("load PGL from PyG has same output get average epsilon", eps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Benchmarking Citation Network')
    parser.add_argument(
        "--dataset", type=str, default="cora", help="dataset (cora, pubmed)")
    parser.add_argument(
        "--feature_pre_normalize",
        type=bool,
        default=True,
        help="pre_normalize feature")
    args = parser.parse_args()
    main(args)
