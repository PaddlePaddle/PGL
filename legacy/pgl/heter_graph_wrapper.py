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
"""
This package provides interface to help building static computational graph
for PaddlePaddle.
"""

import warnings
import numpy as np
import paddle.fluid as fluid

from pgl.utils import op
from pgl.utils import paddle_helper
from pgl.utils.logger import log
from pgl.graph_wrapper import GraphWrapper

ALL = "__ALL__"
__all__ = ["HeterGraphWrapper"]


def is_all(arg):
    """is_all
    """
    return isinstance(arg, str) and arg == ALL


class HeterGraphWrapper(object):
    """Implement a heterogeneous graph wrapper that creates a graph data holders
    that attributes and features in the heterogeneous graph.
    And we provide interface :code:`to_feed` to help converting :code:`Graph`
    data into :code:`feed_dict`.

    Args:
        name: The heterogeneous graph data prefix

        node_feat: A dict of list of tuples that decribe the details of node
                   feature tenosr. Each tuple mush be (name, shape, dtype)
                   and the first dimension of the shape must be set unknown
                   (-1 or None) or we can easily use :code:`HeterGraph.node_feat_info()`
                   to get the node_feat settings.

        edge_feat: A dict of list of tuples that decribe the details of edge
                   feature tenosr. Each tuple mush be (name, shape, dtype)
                   and the first dimension of the shape must be set unknown
                   (-1 or None) or we can easily use :code:`HeterGraph.edge_feat_info()`
                   to get the edge_feat settings.
                   
    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np
            from pgl import heter_graph
            from pgl import heter_graph_wrapper
            num_nodes = 4
            node_types = [(0, 'user'), (1, 'item'), (2, 'item'), (3, 'user')]
            edges = {
                'edges_type1': [(0,1), (3,2)],
                'edges_type2': [(1,2), (3,1)],
            }
            node_feat = {'feature': np.random.randn(4, 16)}
            edges_feat = {
                'edges_type1': {'h': np.random.randn(2, 16)},
                'edges_type2': {'h': np.random.randn(2, 16)},
            }

            g = heter_graph.HeterGraph(
                            num_nodes=num_nodes,
                            edges=edges,
                            node_types=node_types,
                            node_feat=node_feat,
                            edge_feat=edges_feat)
           
            gw = heter_graph_wrapper.HeterGraphWrapper(
                                name='heter_graph', 
                                edge_types = g.edge_types_info(),
                                node_feat=g.node_feat_info(),
                                edge_feat=g.edge_feat_info())
    """

    def __init__(self, name, edge_types, node_feat={}, edge_feat={}, **kwargs):
        self.__data_name_prefix = name
        self._edge_types = edge_types
        self._multi_gw = {}
        for edge_type in self._edge_types:
            type_name = self.__data_name_prefix + '/' + edge_type
            if node_feat:
                n_feat = node_feat
            else:
                n_feat = {}

            if edge_feat:
                e_feat = edge_feat[edge_type]
            else:
                e_feat = {}

            self._multi_gw[edge_type] = GraphWrapper(
                name=type_name,
                node_feat=n_feat,
                edge_feat=e_feat)

    def to_feed(self, heterGraph, edge_types_list=ALL):
        """Convert the graph into feed_dict.

        This function helps to convert graph data into feed dict
        for :code:`fluid.Excecutor` to run the model.

        Args:
            heterGraph: the :code:`HeterGraph` data object
            edge_types_list: the edge types list to be fed

        Return:
            A dictinary contains data holder names and its coresponding data.
        """
        multi_graphs = heterGraph._multi_graph
        if is_all(edge_types_list):
            edge_types_list = self._edge_types

        feed_dict = {}
        for edge_type in edge_types_list:
            feed_d = self._multi_gw[edge_type].to_feed(multi_graphs[edge_type])
            feed_dict.update(feed_d)

        return feed_dict

    def __getitem__(self, edge_type):
        """__getitem__
        """
        return self._multi_gw[edge_type]
