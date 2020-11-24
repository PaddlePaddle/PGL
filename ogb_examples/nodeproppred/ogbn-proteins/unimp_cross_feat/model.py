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
'''build label embedding model
'''
import math
import paddle.fluid as F
import paddle.fluid.layers as L
from module.transformer_gat_pgl import transformer_gat_pgl, split_gw


def linear(input, hidden_size, name, with_bias=True):
    """linear"""
    fan_in = input.shape[-1]
    bias_bound = 1.0 / math.sqrt(fan_in)
    if with_bias:
        fc_bias_attr = F.ParamAttr(
            initializer=F.initializer.UniformInitializer(
                low=-bias_bound, high=bias_bound))
    else:
        fc_bias_attr = False

    negative_slope = math.sqrt(5)
    gain = math.sqrt(2.0 / (1 + negative_slope**2))
    std = gain / math.sqrt(fan_in)
    weight_bound = math.sqrt(3.0) * std
    fc_w_attr = F.ParamAttr(initializer=F.initializer.UniformInitializer(
        low=-weight_bound, high=weight_bound))

    output = L.fc(input,
                  hidden_size,
                  param_attr=fc_w_attr,
                  name=name,
                  bias_attr=fc_bias_attr)
    return output


class Proteins_baseline_model():
    def __init__(self, gw, hidden_size, num_heads, dropout, num_layers):
        '''Proteins_baseline_model
        '''
        self.gw = gw
        self.split_gw = split_gw(gw, 5)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = num_layers
        self.out_size = 112
        self.embed_size = hidden_size * num_heads
        self.build_model()

    def embed_input(self, feature, name, norm=True):
        fan_in = feature.shape[-1]
        bias_bound = 1.0 / math.sqrt(fan_in)
        fc_bias_attr = F.ParamAttr(
            initializer=F.initializer.UniformInitializer(
                low=-bias_bound, high=bias_bound))

        negative_slope = math.sqrt(5)
        gain = math.sqrt(2.0 / (1 + negative_slope**2))
        std = gain / math.sqrt(fan_in)
        weight_bound = math.sqrt(3.0) * std
        fc_w_attr = F.ParamAttr(initializer=F.initializer.UniformInitializer(
            low=-weight_bound, high=weight_bound))

        feature = L.fc(feature,
                       self.embed_size,
                       act=None,
                       param_attr=fc_w_attr,
                       bias_attr=fc_bias_attr,
                       name=name + "_node_feature_encoder")

        if norm:
            lay_norm_attr = F.ParamAttr(
                initializer=F.initializer.ConstantInitializer(value=1))
            lay_norm_bias = F.ParamAttr(
                initializer=F.initializer.ConstantInitializer(value=0))
            feature = L.layer_norm(
                feature,
                name=name + '_layer_norm_feature_input',
                param_attr=lay_norm_attr,
                bias_attr=lay_norm_bias)

        return feature

    def get_gat_layer(self,
                      i,
                      gw,
                      feature,
                      hidden_size,
                      num_heads,
                      concat=True,
                      layer_norm=True,
                      relu=True,
                      gate=False,
                      edge_feature=None):

        fan_in = feature.shape[-1]
        bias_bound = 1.0 / math.sqrt(fan_in)
        fc_bias_attr = F.ParamAttr(
            initializer=F.initializer.UniformInitializer(
                low=-bias_bound, high=bias_bound))

        negative_slope = math.sqrt(5)
        gain = math.sqrt(2.0 / (1 + negative_slope**2))
        std = gain / math.sqrt(fan_in)
        weight_bound = math.sqrt(3.0) * std
        fc_w_attr = F.ParamAttr(initializer=F.initializer.UniformInitializer(
            low=-weight_bound, high=weight_bound))

        if concat:
            skip_feature = L.fc(feature,
                                hidden_size * num_heads,
                                param_attr=fc_w_attr,
                                name='fc_skip_' + str(i),
                                bias_attr=fc_bias_attr)
        else:
            skip_feature = L.fc(feature,
                                hidden_size,
                                param_attr=fc_w_attr,
                                name='fc_skip_' + str(i),
                                bias_attr=fc_bias_attr)
        out_feat, checkpoints = transformer_gat_pgl(
            self.split_gw,
            feature,
            hidden_size,
            'gat_' + str(i),
            num_heads,
            concat=concat,
            edge_feature=edge_feature)

        if gate:
            fan_in = out_feat.shape[-1] * 3
            bias_bound = 1.0 / math.sqrt(fan_in)
            fc_bias_attr = F.ParamAttr(
                initializer=F.initializer.UniformInitializer(
                    low=-bias_bound, high=bias_bound))

            negative_slope = math.sqrt(5)
            gain = math.sqrt(2.0 / (1 + negative_slope**2))
            std = gain / math.sqrt(fan_in)
            weight_bound = math.sqrt(3.0) * std
            fc_w_attr = F.ParamAttr(
                initializer=F.initializer.UniformInitializer(
                    low=-weight_bound, high=weight_bound))

            gate_f = L.fc([skip_feature, out_feat, out_feat - skip_feature],
                          1,
                          param_attr=fc_w_attr,
                          name='gate_' + str(i),
                          bias_attr=fc_bias_attr)

            gate_f = L.sigmoid(gate_f)

            out_feat = skip_feature * gate_f + out_feat * (1 - gate_f)

        else:
            out_feat = out_feat + skip_feature

        if layer_norm:
            lay_norm_attr = F.ParamAttr(
                initializer=F.initializer.ConstantInitializer(value=1))
            lay_norm_bias = F.ParamAttr(
                initializer=F.initializer.ConstantInitializer(value=0))
            out_feat = L.layer_norm(
                out_feat,
                name='layer_norm_' + str(i),
                param_attr=lay_norm_attr,
                bias_attr=lay_norm_bias)
        if relu:
            out_feat = L.relu(out_feat)
        return out_feat

    def build_model(self):
        self.checkpoints = []

        feature_batch = self.embed_input(
            self.gw.node_feat['feat'], name="node")

        def edge_func(efeat):
            efeat = self.embed_input(efeat, name="edge", norm=False)
            return efeat

        edge_feature = [[("edge", edge_func(gw.edge_feat["feat"]))]
                        for gw in self.split_gw]

        self.checkpoints.append(feature_batch)

        for i in range(self.num_layers):
            feature_batch = self.get_gat_layer(
                i,
                self.gw,
                feature_batch,
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                concat=True,
                edge_feature=edge_feature,
                layer_norm=True,
                relu=True)
            if self.dropout > 0:
                feature_batch = L.dropout(
                    feature_batch,
                    dropout_prob=self.dropout,
                    dropout_implementation='upscale_in_train')
            self.checkpoints.append(feature_batch)

        feature_batch = linear(
            feature_batch, hidden_size=self.out_size, name="final")

        self.out_feat = feature_batch

    def train_program(self, ):
        label = F.data(
            name="label", shape=[None, self.out_size], dtype="int64")
        train_idx = F.data(name='train_idx', shape=[None], dtype="int64")
        prediction = L.gather(self.out_feat, train_idx, overwrite=False)
        label = L.gather(label, train_idx, overwrite=False)
        label = L.cast(label, dtype="float32")
        cost = L.sigmoid_cross_entropy_with_logits(x=prediction, label=label)
        avg_cost = L.mean(cost)
        self.avg_cost = avg_cost


class Proteins_label_embedding_model(Proteins_baseline_model):
    def __init__(self, gw, hidden_size, num_heads, dropout, num_layers):
        '''Proteins_label_embedding_model
        '''
        super(Proteins_label_embedding_model, self).__init__(
            gw=gw,
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=num_layers)

    def label_embed_input(self, feature):
        label = F.data(
            name="label", shape=[None, self.out_size], dtype="int64")
        label_idx = F.data(name='label_idx', shape=[None], dtype="int64")
        label = L.gather(label, label_idx, overwrite=False)
        label = L.cast(label, dtype="float32")

        label_feat = self.embed_input(label, "label_feat")
        feature_label = L.gather(feature, label_idx, overwrite=False)

        feature_label = feature_label + label_feat
        feature = L.scatter(feature, label_idx, feature_label, overwrite=True)
        return feature

    def build_model(self):
        self.checkpoints = []

        label_feature = self.embed_input(
            self.gw.node_feat['feat'], name="node")
        label_feature = self.label_embed_input(label_feature)

        def edge_func(efeat):
            efeat = self.embed_input(efeat, name="edge", norm=False)
            return efeat

        edge_feature = [[("edge", edge_func(gw.edge_feat["feat"]))]
                        for gw in self.split_gw]

        feature_batch = label_feature
        self.checkpoints.append(feature_batch)
        for i in range(self.num_layers):
            feature_batch = self.get_gat_layer(
                i,
                self.gw,
                feature_batch,
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                concat=True,
                edge_feature=edge_feature,
                layer_norm=True,
                relu=True)
            if self.dropout > 0:
                feature_batch = L.dropout(
                    feature_batch,
                    dropout_prob=self.dropout,
                    dropout_implementation='upscale_in_train')
            self.checkpoints.append(feature_batch)

        feature_batch = linear(
            feature_batch, hidden_size=self.out_size, name="final")

        self.out_feat = feature_batch
