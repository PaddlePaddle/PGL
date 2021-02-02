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

import paddle.fluid.layers as L
import numpy as np


def self_attention_and_residual(feature, size, input_mask, name, maxlen):
    query = L.fc(feature, size, name=name + "_query", num_flatten_dims=2)
    key = L.fc(feature, size, name=name + "_key", num_flatten_dims=2)
    value = L.fc(feature, size, name=name + "_value", num_flatten_dims=2)
    attention = L.softmax(L.matmul(query, key, transpose_y=True) + input_mask)
    output = L.matmul(attention, value)
    output = L.fc(output, size, name=name + "_model", num_flatten_dims=2)
    output = L.relu(output + feature)
    output = L.layer_norm(output, begin_norm_axis=2, name=name + '_ln')
    return output


def cross_edge_feat(graph_wrapper,
                    feat,
                    hidden_size,
                    num_layers=3,
                    max_neigh=64):
    if num_layers == 0:
        return None

    def send_func(src, dst, efeat):
        return efeat["h"]

    def recv_func(msg):
        pad_value = L.assign(input=np.array([0.0], dtype=np.float32))

        output, length = L.sequence_pad(msg, pad_value, maxlen=max_neigh)
        mask = L.sequence_mask(length, dtype="float32", maxlen=max_neigh)
        mask = L.unsqueeze(mask, [2])
        input_mask = (L.matmul(mask, mask, transpose_y=True) - 1) * -10000
        for layer in range(num_layers):
            output = self_attention_and_residual(
                output,
                hidden_size,
                input_mask,
                name="cross_feat_%s" % layer,
                maxlen=max_neigh)
        return L.reduce_sum(output * mask, 1) / L.reduce_sum(mask, 1)

    feat = L.fc(feat, size=hidden_size, name="init_edge_feat")
    msg = graph_wrapper.send(send_func, efeat_list=[("h", feat)])
    outputs = graph_wrapper.recv(msg, recv_func)
    return outputs
