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
"""This package implements common layers to help building
graph neural networks.
"""

__all__ = ["SAGPooling"]
import pgl
import math
import paddle
import paddle.nn as nn
from pgl.nn import GCNConv
from pgl.math import segment_sum, segment_softmax
from pgl.utils.transform import filter_adj


class SAGPooling(nn.Layer):
    """Implementation of Graph Pooling "SAGPool"
        
    Reference Paper: Self-Attention Graph Pooling

    Args:
        input_dim: The size of the inputs.
        ratio: The ratio to reserve topk nodes.
        GNN: Graph Convolution Operation to generate score for every node. Default:None
        min_score: if min_score is not None, the operator only remove 
                            the node with value lower than min_score. Default:None
        nonlinearity: a nonlinear transform on score when min_score is not None. Default:None 

    """

    def __init__(self,
                 input_dim,
                 ratio=0.5,
                 GNN=None,
                 min_score=None,
                 nonlinearity=None):
        super(SAGPooling, self).__init__()
        self.input_dim = input_dim
        self.ratio = ratio
        GNN = GCNConv if GNN is None else GNN
        self.gnn = GNN(input_dim, 1)
        self.min_score = min_score
        self.nonlinearity = paddle.tanh if nonlinearity is None else nonlinearity

    def forward(self, graph, x):
        """
        Args:          
            graph: `pgl.Graph` instance.
            x: A tensor with shape (num_nodes, input_size).
        Return:
            x: Reserved node features with shape (num_nodes * ratio, feature_size).
            batch: the updated graph_node_id.
            g:  updated 'pgl.Graph' instance (pgl.Graph).
        """
        batch = graph.graph_node_id
        attn_score = self.gnn(graph, x).reshape([-1])
        if self.min_score is None:
            attn_score = self.nonlinearity(attn_score)
        else:
            attn_score = segment_softmax(attn_score, batch)
        rank = segment_topk(attn_score, self.ratio, batch, self.min_score)
        x = x[rank] * attn_score[rank].reshape([-1, 1])
        batch = batch[rank]
        edge_index, edge_attr = filter_adj(
            graph.edges, rank, num_nodes=attn_score.shape[0])
        batch_size = graph.num_graph
        num_nodes = segment_sum(paddle.ones([x.shape[0]]), batch)
        graph_node_index = paddle.scatter(
            paddle.zeros([batch_size + 1]).astype(paddle.float32),
            paddle.arange(1, batch_size + 1),
            num_nodes.cumsum(0).astype(paddle.float32)).astype(paddle.int64)
        g = pgl.Graph(
            num_nodes=x.shape[0],
            edges=edge_index.transpose([1, 0]),
            node_feat={"attr": x},
            _graph_node_index=graph_node_index,
            _num_graph=(batch.max() + 1).item())

        # g = g.tensor()
        return x, batch, g


def segment_topk(x, ratio, segment_ids, min_score=None):
    """
    Segment topk operator.

    This operator select the topk value of input elements which with the same index in 'segment_ids' ,
    and return the index of reserved nodes into with shape [max_num_nodes*ratio, dim].
    
    if min_score is not None, the operator only remove the node with value lower that min_score
    
    Args:
        x (tensor): a tensor, available data type float32, float64.
        segment_ids (tensor): a 1-d tensor, which have the same size
                            with the first dimension of input data.
                            available data type is int32, int64.
        ratio (float): a ratio that reserving present nodes
        min_score (float): if min_score is not None, the operator only remove 
                            the node with value lower than min_score

    Returns:
        perm (Tensor): the index of reserved nodes
    """
    if min_score is not None:
        scores_max = math.segment_max(x, segment_ids).index_select(segment_ids,
                                                                   0) - tol
        scores_min = scores_max.clip(max=min_score)
        perm = (x > scores_min).reshape([-1]).nonzero(as_tuple=False).reshape(
            [-1])
    else:
        num_nodes = segment_sum(paddle.ones([x.shape[0]]), segment_ids)
        batch_size, max_num_nodes = int(num_nodes.shape[0]), int(num_nodes.max(
        ).item())
        if num_nodes.cumsum(0).shape[0] > 1:
            cum_num_nodes = paddle.concat(
                [paddle.zeros([1]), num_nodes.cumsum(0)[:-1]], 0)
        else:
            cum_num_nodes = paddle.concat([paddle.zeros([1])], 0)
        index = paddle.arange(segment_ids.shape[0], dtype=paddle.int32)
        index = (index - cum_num_nodes[segment_ids]) + (segment_ids *
                                                        max_num_nodes)
        dense_x = paddle.full([batch_size * max_num_nodes],
                              -1e20).astype(paddle.float32)
        dense_x = paddle.scatter(dense_x, index, x.reshape([-1]))
        # dense_x[index] = x
        dense_x = dense_x.reshape([batch_size, max_num_nodes])
        perm = dense_x.argsort(-1, descending=True)
        perm = perm + cum_num_nodes.reshape([-1, 1])
        perm = perm.reshape([-1])
        if isinstance(ratio, int):
            k = paddle.full([num_nodes.shape[0]], ratio)
            k = paddle.min(k, num_nodes)
        else:
            k = (ratio *
                 num_nodes.astype(paddle.float32)).ceil().astype(paddle.int64)
        mask = [
            paddle.arange(
                k[i],
                dtype=paddle.int64, ) + i * max_num_nodes
            for i in range(batch_size)
        ]
        perm = paddle.concat([perm[i] for i in mask], axis=0)
    return perm
