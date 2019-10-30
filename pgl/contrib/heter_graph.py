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
    This package implement Heterogeneous Graph structure for handling Heterogeneous graph data.
"""
import time
import numpy as np
import pickle as pkl
import time
import pgl.graph_kernel as graph_kernel
from pgl.graph import Graph

__all__ = ['HeterGraph']


def _hide_num_nodes(shape):
    """Set the first dimension as unknown
    """
    shape = list(shape)
    shape[0] = None
    return shape


class NodeGraph(Graph):
    """Implementation of a graph that has multple node types.

    Args:
        num_nodes: number of nodes in the graph
        edges: list of (u, v) tuples
        node_types (optional): list of (u, node_type) tuples to specify the node type of every node
        node_feat (optional): a dict of numpy array as node features
        edge_feat (optional): a dict of numpy array as edge features
    """

    def __init__(self,
                 num_nodes,
                 edges,
                 node_types=None,
                 node_feat=None,
                 edge_feat=None):
        super(NodeGraph, self).__init__(num_nodes, edges, node_feat, edge_feat)

        if isinstance(node_types, list):
            self._node_types = np.array(node_types, dtype=object)[:, 1]
        else:
            self._node_types = node_types


class HeterGraph(object):
    """Implementation of heterogeneous graph structure in pgl

    This is a simple implementation of heterogeneous graph structure in pgl.

    Args:
        num_nodes: number of nodes in a heterogeneous graph
        edges: dict, every element in dict is a list of (u, v) tuples.
        node_types (optional): list of (u, node_type) tuples to specify the node type of every node
        node_feat (optional): a dict of numpy array as node features
        edge_feat (optional): a dict of dict as edge features for every edge type

    Examples:
        .. code-block:: python

            import numpy as np
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
    """

    def __init__(self,
                 num_nodes,
                 edges,
                 node_types=None,
                 node_feat=None,
                 edge_feat=None):
        self._num_nodes = num_nodes
        self._edges_dict = edges

        if node_feat is not None:
            self._node_feat = node_feat
        else:
            self._node_feat = {}

        if edge_feat is not None:
            self._edge_feat = edge_feat
        else:
            self._edge_feat = {}

        self._multi_graph = {}
        for key, value in self._edges_dict.items():
            if not self._edge_feat:
                edge_feat = None
            else:
                edge_feat = self._edge_feat[key]

            self._multi_graph[key] = NodeGraph(
                num_nodes=self._num_nodes,
                edges=value,
                node_types=node_types,
                node_feat=self._node_feat,
                edge_feat=edge_feat)

    @property
    def num_nodes(self):
        """Return the number of nodes.
        """
        return self._num_nodes

    def __getitem__(self, edge_type):
        """__getitem__
        """
        return self._multi_graph[edge_type]

    def node_feat_info(self):
        """Return the information of node feature for HeterGraphWrapper.

        This function return the information of node features of all node types. And this
        function is used to help constructing HeterGraphWrapper

        Return:
            A list of tuple (name, shape, dtype) for all given node feature.

        """
        node_feat_info = []
        for feat_name, feat in self._node_feat.items():
            node_feat_info.append(
                (feat_name, _hide_num_nodes(feat.shape), feat.dtype))

        return node_feat_info

    def edge_feat_info(self):
        """Return the information of edge feature for HeterGraphWrapper.

        This function return the information of edge features of all edge types. And this
        function is used to help constructing HeterGraphWrapper

        Return:
            A dict of list of tuple (name, shape, dtype) for all given edge feature.

        """
        edge_feat_info = {}
        for edge_type_name, feat_dict in self._edge_feat.items():
            tmp_edge_feat_info = []
            for feat_name, feat in feat_dict.items():
                full_name = feat_name
                tmp_edge_feat_info.append(
                    (full_name, _hide_num_nodes(feat.shape), feat.dtype))
            edge_feat_info[edge_type_name] = tmp_edge_feat_info
        return edge_feat_info

    def edge_types_info(self):
        """Return the information of all edge types.
        
        Return:
            A list of all edge types.
        
        """
        edge_types_info = []
        for key, _ in self._edges_dict.items():
            edge_types_info.append(key)

        return edge_types_info
