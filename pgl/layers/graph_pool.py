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
"""This package implements common layers to help building
graph neural networks.
"""
import paddle.fluid as fluid
from pgl import graph_wrapper
from pgl.utils import paddle_helper
from pgl.utils import op

__all__ = ['graph_pooling']


def graph_pooling(gw, node_feat, pool_type):
    """Implementation of graph pooling 

    This is an implementation of graph pooling

    Args:
        gw: Graph wrapper object (:code:`StaticGraphWrapper` or :code:`GraphWrapper`)

        node_feat: A tensor with shape (num_nodes, feature_size).

        pool_type: The type of pooling ("sum", "average" , "min")

    Return:
        A tensor with shape (num_graph, hidden_size)
    """
    graph_feat = op.nested_lod_reset(node_feat, gw.graph_lod)
    graph_feat = fluid.layers.sequence_pool(graph_feat, pool_type)
    return graph_feat
