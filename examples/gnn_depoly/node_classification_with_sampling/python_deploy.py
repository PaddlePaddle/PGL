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
from functools import partial

import paddle
import paddle.nn as nn
import pgl
from paddle.io import Dataset, DataLoader

from dataset import ShardedDataset
from dataset import generate_batch_infer_data
from train import load_reddit_data
from train import get_basic_graph_sample_neighbors_info
from train import get_sample_graph_list


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
        description='Node Classification with graph sampling, python prediction.'
    )
    parser.add_argument(
        "--model", type=str, default="GraphSage", help="GraphSage, GAT")
    parser.add_argument("--model_dir", type=str, default="./export")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--samples", nargs='+', type=int, default=[25, 10])
    args = parser.parse_args()

    predictor = Predictor(args.model, args.model_dir)

    graph, indexes, labels, feature = load_reddit_data()
    test_index = indexes[2]
    test_label = labels[2]

    data = [(test_index[idx:idx + args.batch_size],
             test_label[idx:idx + args.batch_size])
            for idx in range(0, len(test_index), args.batch_size)]
    batches = [
        generate_batch_infer_data(data[idx], graph, args.samples)
        for idx in range(len(data))
    ]

    total_correct = total_sample = 0
    for batch in batches:
        g, sample_index, index, label = batch
        logits = predictor.predict(g.edges,
                                   np.array([g.num_nodes]),
                                   feature[sample_index])
        pred = logits[index]
        idx = np.argmax(pred, axis=1)
        correct_num = np.sum(idx == label)
        total_correct += correct_num
        total_sample += index.shape[0]

    print("Test acc: %f" % (total_correct / total_sample))
