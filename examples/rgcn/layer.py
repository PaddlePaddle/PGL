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

import math
import numpy as np
import paddle.fluid as fluid

def RGCNConv(graph_wrapper, in_dim, out_dim, etypes, num_nodes, num_bases=0):
    """Implementation of Relational Graph Convolutional Networks (R-GCN)

    This is an implementation of the paper 
    Modeling Relational Data with Graph Convolutional Networks 
    (http://arxiv.org/abs/1703.06103).

    """
    param_attr_init = fluid.initializer.Uniform(
            low=-1.0, high=1.0, seed=np.random.randint(100))
    tn_initializer = fluid.initializer.TruncatedNormalInitializer(
            loc=0.0, scale=1.0 / math.sqrt(in_dim))
    
    def __message(src_feat, dst_feat, edge_feat):
        """
        send head信息
        """
        return src_feat["h"]
    
    def __reduce(msg):
        """
        sum pool
        """
        return fluid.layers.sequence_pool(msg, pool_type='sum')
    
    nfeat = fluid.layers.create_parameter(shape=[num_nodes, in_dim],
                                        default_initializer=param_attr_init,
                                        dtype='float32',
                                        name='nfeat')
    
    num_rels = len(etypes)
    if num_bases <= 0 or num_bases >= num_rels:
        num_bases = num_rels

    weight = fluid.layers.create_parameter(shape=[num_bases, in_dim, out_dim],
                                        default_initializer=tn_initializer,
                                        dtype='float32',
                                        name='trans_weight')

    if num_bases < num_rels:
        w_comp = fluid.layers.create_parameter(shape=[num_rels, num_bases],
                                        default_initializer=tn_initializer,
                                        dtype='float32',
                                        name='trans_w_comp')
        weight = fluid.layers.transpose(weight, perm=[1, 0, 2])
        weight = fluid.layers.matmul(w_comp, weight)
        # [num_rels, in_dim, out_dim]
        weight = fluid.layers.transpose(weight, perm=[1, 0, 2])

    feat_list = []
    for idx, etype in enumerate(etypes):
        w = weight[idx, :, :]

        h = fluid.layers.matmul(nfeat, w) 
        message = graph_wrapper[etype].send(__message, nfeat_list=[('h', h)])
        output = graph_wrapper[etype].recv(message, __reduce)
        feat_list.append(output)
        
    h = fluid.layers.sum(feat_list)
    return h
