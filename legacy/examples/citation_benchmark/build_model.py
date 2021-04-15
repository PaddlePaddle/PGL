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
import model
from pgl import data_loader
import paddle.fluid as fluid
import numpy as np
import time


def build_model(dataset, config, phase, main_prog):
    gw = pgl.graph_wrapper.GraphWrapper(
        name="graph", node_feat=dataset.graph.node_feat_info())

    GraphModel = getattr(model, config.model_name)
    m = GraphModel(config=config, num_class=dataset.num_classes)
    logits = m.forward(gw, gw.node_feat["words"], phase)

    # Take the last
    node_index = fluid.layers.data(
        "node_index", shape=[None, 1], dtype="int64", append_batch_size=False)
    node_label = fluid.layers.data(
        "node_label", shape=[None, 1], dtype="int64", append_batch_size=False)

    pred = fluid.layers.gather(logits, node_index)
    loss, pred = fluid.layers.softmax_with_cross_entropy(
        logits=pred, label=node_label, return_softmax=True)
    acc = fluid.layers.accuracy(input=pred, label=node_label, k=1)
    loss = fluid.layers.mean(loss)

    if phase == "train":
        adam = fluid.optimizer.Adam(
            learning_rate=config.learning_rate,
            regularization=fluid.regularizer.L2DecayRegularizer(
                regularization_coeff=config.weight_decay))
        adam.minimize(loss)
    return gw, loss, acc
