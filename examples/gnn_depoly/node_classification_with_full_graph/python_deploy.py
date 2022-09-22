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

import argparse
import os
import sys
import time
import numpy as np

import paddle
import paddle.nn as nn
import pgl


class Predictor(object):
    def __init__(self, model, model_dir, device="gpu"):

        args.model = model
        model_file = model_dir + "/inference_%s.pdmodel" % model.lower()
        params_file = model_dir + "/inference_%s.pdiparams" % model.lower()

        config = paddle.inference.Config(model_file, params_file)

        config.switch_use_feed_fetch_ops(False)
        self.predictor = paddle.inference.create_predictor(config)
        self.input_handles = [
            self.predictor.get_input_handle(name)
            for name in self.predictor.get_input_names()
        ]
        self.output_handle = self.predictor.get_output_handle(
            self.predictor.get_output_names()[0])

    def predict(self, edges, num_nodes, feature):

        if args.model == "GraphSage":
            self.input_handles[0].copy_from_cpu(edges)
            self.input_handles[1].copy_from_cpu(feature)
        else:
            self.input_handles[0].copy_from_cpu(edges)
            self.input_handles[1].copy_from_cpu(num_nodes)
            self.input_handles[2].copy_from_cpu(feature)
        self.predictor.run()

        logits = self.output_handle.copy_to_cpu()
        return logits


def normalize(feat):
    return feat / np.maximum(np.sum(feat, -1, keepdims=True), 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Node Classification, no need sampling, python prediction.')
    parser.add_argument(
        "--model", type=str, default="GCN", help="GCN, GAT, GraphSage")
    parser.add_argument("--model_dir", type=str, default="./export")
    args = parser.parse_args()

    dataset = pgl.dataset.CoraDataset()
    dataset.graph.node_feat["words"] = normalize(dataset.graph.node_feat[
        "words"])
    dataset.test_label = dataset.y[dataset.test_index]

    predictor = Predictor(args.model, args.model_dir)

    logits = predictor.predict(dataset.graph.edges,
                               np.array([dataset.graph.num_nodes]),
                               dataset.graph.node_feat["words"])
    pred = logits[dataset.test_index]

    idx = np.argmax(pred, axis=1)
    print("Test acc: %f" % (np.sum(idx == dataset.test_label) / idx.shape[0]))
