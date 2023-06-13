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

import paddle
import paddle.nn as nn

import pgl
import pgl.math as math

__all__ = [
    "degree_norm",
    "graph_pool",
    "graph_norm",
    "edge_softmax",
]


def degree_norm(graph, mode="indegree"):
    """Calculate the degree normalization of a graph

    Args:
        graph: the graph object from (:code:`Graph`)

        mode: which degree to be normalized ("indegree" or "outdegree")

    return:
        A tensor with shape (num_nodes, 1).

    """

    assert mode in [
        'indegree', 'outdegree'
    ], "The degree_norm mode should be in ['indegree', 'outdegree']. But recieve mode=%s" % mode

    if mode == "indegree":
        degree = graph.indegree()
    elif mode == "outdegree":
        degree = graph.outdegree()

    norm = paddle.cast(degree, dtype=paddle.get_default_dtype())
    norm = paddle.clip(norm, min=1.0)
    norm = paddle.pow(norm, -0.5)
    norm = paddle.reshape(norm, [-1, 1])
    return norm


def graph_pool(graph, feature, pool_type):
    """Implementation of graph pooling

    This is an implementation of graph pooling

    Args:
        graph: the graph object from (:code:`Graph`)

        feature: A tensor with shape (num_nodes, feature_size).

        pool_type: The type of pooling ("sum", "mean" , "min", "max")

    Return:
        A tensor with shape (num_graph, feature_size)
    """

    graph_feat = math.segment_pool(feature, graph.graph_node_id, pool_type)
    return graph_feat


def graph_norm(graph, feature):
    """Implementation of graph normalization
   
    Reference Paper: BENCHMARKING GRAPH NEURAL NETWORKS
   
    Each node features is divied by sqrt(num_nodes) per graphs.

    Args:
        graph: the graph object from (:code:`Graph`)

        feature: A tensor with shape (num_nodes, feature_size).

    Return:
        A tensor with shape (num_nodes, hidden_size)
    """

    nodes = paddle.ones(shape=[graph.num_nodes, 1], dtype="float32")
    norm = graph_pool(graph, nodes, pool_type="sum")
    norm = paddle.sqrt(norm)
    norm = paddle.gather(norm, graph.graph_node_id)
    return feature / norm


def edge_softmax(graph, logits, norm_by="dst"):
    """Compute softmax over edge weight of incoming edges of every node.

    Args:

        logits (Tensor): The input edge feature.

        norm_by (str): Normalized by source nodes or destination nodes.
                       ("src" or "dst", default is "dst")

    Return:

        A paddle.Tensor of return softmax value.

    """

    src, dst, eid = graph.sorted_edges(sort_by=norm_by)
    uniq_ind, segment_ids = graph.get_segment_ids(src, dst, segment_by=norm_by)
    logits = paddle.gather(logits, eid, axis=0)
    score = pgl.math.segment_softmax(logits, segment_ids)
    init_output = paddle.zeros_like(score, dtype=score.dtype)
    score = paddle.scatter(init_output, eid, score)
    return score
