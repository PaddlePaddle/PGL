# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
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

from random import random
import numpy as np

import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as L
import pgl
from pgl.graph import Graph, MultiGraph
from pgl.graph_wrapper import GraphWrapper
from pgl.utils.logger import log
from pgl.layers.conv import gcn
from layers import sag_pool
from conv import norm_gcn

class GlobalModel(object):
    """Implementation of global pooling architecture with SAGPool.
    """
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset
        self.hidden_size = args.hidden_size
        self.num_classes = args.num_classes
        self.num_features = args.num_features
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.batch_size = args.batch_size

        graph_data = []
        g, label = self.dataset[0]
        graph_data.append(g)
        g, label = self.dataset[1]
        graph_data.append(g)

        batch_graph = MultiGraph(graph_data)
        indegree = batch_graph.indegree()
        norm = np.zeros_like(indegree, dtype="float32")
        norm[indegree > 0] = np.power(indegree[indegree > 0], -0.5)
        batch_graph.node_feat["norm"] = np.expand_dims(norm, -1)
        graph_data = batch_graph

        self.graph_wrapper = GraphWrapper(
            name="graph",
            node_feat=graph_data.node_feat_info()
            )
        self.labels = L.data(
            "labels",
            shape=[None, self.args.num_classes],
            dtype="int32",
            append_batch_size=False)

        self.labels_1dim = L.data(
          "labels_1dim",
          shape=[None],
          dtype="int32",
          append_batch_size=False)

        self.graph_id = L.data(
          "graph_id",
          shape=[None],
          dtype="int32",
          append_batch_size=False)

        if self.args.dataset_name == "FRANKENSTEIN":
            self.gcn = gcn
        else:
            self.gcn = norm_gcn
        
        self.build_model()

    def build_model(self):
        node_features = self.graph_wrapper.node_feat["feat"]

        output = self.gcn(gw=self.graph_wrapper, 
                     feature=node_features, 
                     hidden_size=self.hidden_size,
                     activation="relu", 
                     norm=self.graph_wrapper.node_feat["norm"],
                     name="gcn_layer_1")
        output1 = output
        output = self.gcn(gw=self.graph_wrapper, 
                     feature=output, 
                     hidden_size=self.hidden_size,
                     activation="relu", 
                     norm=self.graph_wrapper.node_feat["norm"],
                     name="gcn_layer_2")
        output2 = output
        output = self.gcn(gw=self.graph_wrapper, 
                     feature=output, 
                     hidden_size=self.hidden_size,
                     activation="relu", 
                     norm=self.graph_wrapper.node_feat["norm"],
                     name="gcn_layer_3")
        
        output = L.concat(input=[output1, output2, output], axis=-1)

        output, ratio_length = sag_pool(gw=self.graph_wrapper, 
                          feature=output, 
                          ratio=self.pooling_ratio,
                          graph_id=self.graph_id,
                          dataset=self.args.dataset_name,
                          name="sag_pool_1")
        output = L.lod_reset(output, self.graph_wrapper.graph_lod)
        cat1 = L.sequence_pool(output, "sum")
        ratio_length = L.cast(ratio_length, dtype="float32")
        cat1 = L.elementwise_div(cat1, ratio_length, axis=-1)
        cat2 = L.sequence_pool(output, "max")
        output = L.concat(input=[cat2, cat1], axis=-1)

        output = L.fc(output, size=self.hidden_size, act="relu")
        output = L.dropout(output, dropout_prob=self.dropout_ratio)
        output = L.fc(output, size=self.hidden_size // 2, act="relu")
        output = L.fc(output, size=self.num_classes, act=None,
                      param_attr=fluid.ParamAttr(name="final_fc")) 

        self.labels = L.cast(self.labels, dtype="float32")
        loss = L.sigmoid_cross_entropy_with_logits(x=output, label=self.labels)
        self.loss = L.mean(loss)
        pred = L.sigmoid(output) 
        self.pred = L.argmax(x=pred, axis=-1) 
        correct = L.equal(self.pred, self.labels_1dim)
        correct = L.cast(correct, dtype="int32")
        self.correct = L.reduce_sum(correct)
