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

import paddle.fluid as fluid
import paddle.fluid.layers as L

def norm_gcn(gw, feature, hidden_size, activation, name, norm=None):
    """Implementation of graph convolutional neural networks(GCN), using different 
       normalization method.
    Args:
        gw: Graph wrapper object.

        feature: A tensor with shape (num_nodes, feature_size).

        hidden_size: The hidden size for norm gcn.

        activation: The activation for the output.

        name: Norm gcn layer names.

        norm: If norm is not None, then the feature will be normalized. Norm must
              be tensor with shape (num_nodes,) and dtype float32.

    Return:
        A tensor with shape (num_nodes, hidden_size)
    """

    size = feature.shape[-1]
    feature = L.fc(feature,
                   size=hidden_size,
                   bias_attr=False,
                   param_attr=fluid.ParamAttr(name=name))

    if norm is not None:
        src, dst = gw.edges
        norm_src = L.gather(norm, src, overwrite=False)
        norm_dst = L.gather(norm, dst, overwrite=False)
        norm = norm_src * norm_dst

        def send_src_copy(src_feat, dst_feat, edge_feat):
            return src_feat["h"] * norm
    else:
        def send_src_copy(src_feat, dst_feat, edge_feat):
            return src_feat["h"]

    msg = gw.send(send_src_copy, nfeat_list=[("h", feature)])
    output = gw.recv(msg, "sum")

    bias = L.create_parameter(
        shape=[hidden_size],
        dtype='float32',
        is_bias=True,
        name=name + '_bias')
    output = L.elementwise_add(output, bias, act=activation)
    return output
