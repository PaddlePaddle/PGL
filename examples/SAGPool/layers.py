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

import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as L
import pgl
from pgl.graph_wrapper import GraphWrapper
from pgl.utils.logger import log
from conv import norm_gcn
from pgl.layers.conv import gcn

def topk_pool(gw, score, graph_id, ratio):
    """Implementation of topk pooling, where k means pooling ratio.
    
    Args:
        gw: Graph wrapper object.

        score: The attention score of all nodes, which is used to select 
               important nodes.

        graph_id: The graphs that the nodes belong to.

        ratio: The pooling ratio of nodes we want to select.

    Return: 
        perm: The index of nodes we choose.

        ratio_length: The selected node numbers of each graph.
    """

    graph_lod = gw.graph_lod
    graph_nodes = gw.num_nodes
    num_graph = gw.num_graph

    num_nodes = L.ones(shape=[graph_nodes], dtype="float32")
    num_nodes = L.lod_reset(num_nodes, graph_lod)
    num_nodes_per_graph = L.sequence_pool(num_nodes, pool_type='sum')
    max_num_nodes = L.reduce_max(num_nodes_per_graph, dim=0) 
    max_num_nodes = L.cast(max_num_nodes, dtype="int32")

    index = L.arange(0, gw.num_nodes, dtype="int64")
    offset = L.gather(graph_lod, graph_id, overwrite=False)
    index = (index - offset) + (graph_id * max_num_nodes)
    index.stop_gradient = True
    
    # padding
    dense_score = L.fill_constant(shape=[num_graph * max_num_nodes],
                                  dtype="float32", value=-999999)
    index = L.reshape(index, shape=[-1])
    dense_score = L.scatter(dense_score, index, updates=score)
    num_graph = L.cast(num_graph, dtype="int32")
    dense_score = L.reshape(dense_score, 
                            shape=[num_graph, max_num_nodes])

    # record the sorted index
    _, sort_index = L.argsort(dense_score, axis=-1, descending=True)

    # recover the index range
    graph_lod = graph_lod[:-1]
    graph_lod = L.reshape(graph_lod, shape=[-1, 1])
    graph_lod = L.cast(graph_lod, dtype="int64")
    sort_index = L.elementwise_add(sort_index, graph_lod, axis=-1)
    sort_index = L.reshape(sort_index, shape=[-1, 1])

    # use sequence_slice to choose selected node index
    pad_lod = L.arange(0, (num_graph + 1) * max_num_nodes, step=max_num_nodes, dtype="int32")
    sort_index = L.lod_reset(sort_index, pad_lod)
    ratio_length = L.ceil(num_nodes_per_graph * ratio) 
    ratio_length = L.cast(ratio_length, dtype="int64")
    ratio_length = L.reshape(ratio_length, shape=[-1, 1])
    offset = L.zeros(shape=[num_graph, 1], dtype="int64") 
    choose_index = L.sequence_slice(input=sort_index, offset=offset, length=ratio_length) 

    perm = L.reshape(choose_index, shape=[-1])
    return perm, ratio_length 


def sag_pool(gw, feature, ratio, graph_id, dataset, name, activation=L.tanh):
    """Implementation of self-attention graph pooling (SAGPool)

    This is an implementation of the paper SELF-ATTENTION GRAPH POOLING
    (https://arxiv.org/pdf/1904.08082.pdf)

    Args:
        gw: Graph wrapper object.

        feature: A tensor with shape (num_nodes, feature_size).

        ratio: The pooling ratio of nodes we want to select.

        graph_id: The graphs that the nodes belong to. 

        dataset: To differentiate FRANKENSTEIN dataset and other datasets.

        name: The name of SAGPool layer.
        
        activation: The activation function.

    Return:
        new_feature: A tensor with shape (num_nodes, feature_size), and the unselected
                     nodes' feature is masked by zero.

        ratio_length: The selected node numbers of each graph.

    """
    if dataset == "FRANKENSTEIN":
        gcn_ = gcn
    else:
        gcn_ = norm_gcn

    score = gcn_(gw=gw,    
                feature=feature, 
                hidden_size=1,
                activation=None,
                norm=gw.node_feat["norm"],
                name=name)
    score = L.squeeze(score, axes=[])  
    perm, ratio_length = topk_pool(gw, score, graph_id, ratio) 

    mask = L.zeros_like(score)
    mask = L.cast(mask, dtype="float32")
    updates = L.ones_like(perm)
    updates = L.cast(updates, dtype="float32")
    mask = L.scatter(mask, perm, updates)
    new_feature = L.elementwise_mul(feature, mask, axis=0)
    temp_score = activation(score)
    new_feature = L.elementwise_mul(new_feature, temp_score, axis=0)
    return new_feature, ratio_length 
