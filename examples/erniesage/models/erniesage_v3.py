# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
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
import paddle.fluid as F
import paddle.fluid.layers as L

from models.base import BaseNet, BaseGNNModel
from models.ernie_model.ernie import ErnieModel
from models.ernie_model.ernie import ErnieGraphModel
from models.ernie_model.ernie import ErnieConfig
from models.message_passing import copy_send


class ErnieSageV3(BaseNet):
    def __init__(self, config):
        super(ErnieSageV3, self).__init__(config)
        self.config.layer_type = "ernie_recv_sum"

    def build_inputs(self):
        inputs = super(ErnieSageV3, self).build_inputs()
        term_ids = L.data(
            "term_ids", shape=[None, self.config.max_seqlen], dtype="int64", append_batch_size=False)
        return inputs + [term_ids]

    def gnn_layer(self, gw, feature, hidden_size, act, initializer, learning_rate, name):
        def ernie_recv(feat):
            """doc"""
            # TODO maxlen  400
            #pad_value = L.cast(L.assign(input=np.array([0], dtype=np.int32)), "int64")
            pad_value = L.zeros([1], "int64")
            out, _ = L.sequence_pad(feat, pad_value=pad_value, maxlen=10)
            out = L.reshape(out, [0, 400])
            return out

        def erniesage_v3_aggregator(gw, feature, hidden_size, act, initializer, learning_rate, name):
            msg = gw.send(copy_send, nfeat_list=[("h", feature)])
            neigh_feature = gw.recv(msg, ernie_recv)
            neigh_feature = L.cast(L.unsqueeze(neigh_feature, [-1]), "int64")

            feature = L.unsqueeze(feature, [-1])
            cls = L.fill_constant_batch_size_like(feature, [-1, 1, 1], "int64", 1)
            term_ids = L.concat([cls, feature[:, :-1], neigh_feature], 1)
            term_ids.stop_gradient = True
            return term_ids
        return erniesage_v3_aggregator(gw, feature, hidden_size, act, initializer, learning_rate, name)

    def gnn_layers(self, graph_wrappers, feature):
        features = [feature]

        initializer = None
        fc_lr = self.config.lr / 0.001

        for i in range(self.config.num_layers):
            if i == self.config.num_layers - 1:
                act = None
            else:
                act = "leaky_relu"

            feature = self.gnn_layer(
                graph_wrappers[i],
                feature,
                self.config.hidden_size,
                act,
                initializer,
                learning_rate=fc_lr,
                name="%s_%s" % (self.config.layer_type, i))
            features.append(feature)
        return features

    def take_final_feature(self, feature, index, name):
        """take final feature"""
        feat = L.gather(feature, index, overwrite=False)

        ernie_config = self.config.ernie_config
        ernie = ErnieGraphModel(
            src_ids=feat,
            config=ernie_config,
            slot_seqlen=self.config.max_seqlen,
            name="student_")
        feat = ernie.get_pooled_output()
        fc_lr = self.config.lr / 0.001
        feat= L.fc(feat,
                   self.config.hidden_size,
                   act="relu",
                   param_attr=F.ParamAttr(name=name + "_l",
                   learning_rate=fc_lr),
                   )
        feat = L.l2_normalize(feat, axis=1)

        if self.config.final_fc:
            feat = L.fc(feat,
                           self.config.hidden_size,
                           param_attr=F.ParamAttr(name=name + '_w'),
                           bias_attr=F.ParamAttr(name=name + '_b'))

        if self.config.final_l2_norm:
            feat = L.l2_normalize(feat, axis=1)
        return feat

    def __call__(self, graph_wrappers):
        inputs = self.build_inputs()
        feature = inputs[-1]
        features = self.gnn_layers(graph_wrappers, feature)
        outputs = [self.take_final_feature(features[-1], i, "final_fc") for i in inputs[:-1]]
        src_real_index = L.gather(graph_wrappers[0].node_feat['index'], inputs[0])
        outputs.append(src_real_index)
        return inputs, outputs
    

class ErnieSageModelV3(BaseGNNModel):
    def gen_net_fn(self, config):
        return ErnieSageV3(config)
