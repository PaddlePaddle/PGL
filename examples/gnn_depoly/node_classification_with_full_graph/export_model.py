# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
import argparse

import numpy as np
import paddle
import pgl

sys.path.insert(0, os.path.abspath(".."))
from models import GCN, GAT, GraphSage


def save_static_model(args):
    dataset = pgl.dataset.CoraDataset()

    if args.model == "GAT":
        model = GAT(input_size=dataset.graph.node_feat["words"].shape[1],
                    num_class=dataset.num_classes,
                    num_layers=2,
                    feat_drop=0.6,
                    attn_drop=0.6,
                    num_heads=8,
                    hidden_size=8)
    elif args.model == "GCN":
        model = GCN(input_size=dataset.graph.node_feat["words"].shape[1],
                    num_class=dataset.num_classes,
                    num_layers=1,
                    dropout=0.5,
                    hidden_size=16)
    elif args.model == "GraphSage":
        model = GraphSage(
            input_size=dataset.graph.node_feat["words"].shape[1],
            num_class=dataset.num_classes,
            num_layers=2,
            hidden_size=16)
    else:
        raise ValueError("%s model is supported!" % args.model)
    state_dict = paddle.load("%s.pdparam" % args.model.lower())
    model.set_state_dict(state_dict)
    model.eval()

    # Convert to static graph with specific input description
    if args.model == "GraphSage":
        model = paddle.jit.to_static(
            model,
            input_spec=[
                paddle.static.InputSpec(
                    shape=[None, None], dtype="int64"),  # edge_index
                paddle.static.InputSpec(
                    shape=[None, None], dtype="float32"),  # feature
            ])
    else:
        model = paddle.jit.to_static(
            model,
            input_spec=[
                paddle.static.InputSpec(
                    shape=[None, None], dtype="int64"),  # edge_index
                paddle.static.InputSpec(
                    shape=[None], dtype="int64"),  # num_nodes
                paddle.static.InputSpec(
                    shape=[None, None],
                    dtype="float32")  # feature                
            ])

    # Save in static graph model
    if not os.path.exists("./export"):
        os.mkdir("./export")
    save_path = os.path.join("export", "inference_%s" % args.model.lower())
    paddle.jit.save(model, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Node Classification, no need sampling, export model.")
    parser.add_argument(
        "--model", type=str, default="GCN", help="GCN, GAT, GraphSage")
    args = parser.parse_args()
    save_static_model(args)
