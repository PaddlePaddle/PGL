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
import numpy as np
import pickle as pkl
import time
import pgl.graph_kernel as graph_kernel
from pgl import graph

__all__ = ['HeterGraph']


def _hide_num_nodes(shape):
    """Set the first dimension as unknown
    """
    shape = list(shape)
    shape[0] = None
    return shape


class HeterGraph(object):
    """Implementation of graph structure in pgl

    This is a simple implementation of heterogeneous graph structure in pgl

    Args:
        num_nodes_every_type: dict, number of nodes for every node type

        edges_every_type: dict, every element is a list of (u, v) tuples.

        node_feat_every_type: features for every node type.

    Examples:
        .. code-block:: python

            import numpy as np
            num_nodes_every_type = {'type1':3,'type2':4, 'type3':2}
            edges_every_type = {
                ('type1','type2', 'edges_type1'): [(0,1), (1,2)], 
                ('type1', 'type3', 'edges_type2'): [(1,2), (3,1)],
            }
            node_feat_every_type = {
                'type1': {'features1': np.random.randn(3, 4),
                          'features2': np.random.randn(3, 4)}, 
                'type2': {'features3': np.random.randn(4, 4)},
                'type3': {'features1': np.random.randn(2, 4),
                          'features2': np.random.randn(2, 4)}
            }
            edges_feat_every_type = {
                ('type1','type2','edges_type1'): {'h': np.random.randn(2, 4)},
                ('type1', 'type3', 'edges_type2'): {'h':np.random.randn(2, 4)},
            }

            g = heter_graph.HeterGraph(
                            num_nodes_every_type=num_nodes_every_type, 
                            edges_every_type=edges_every_type,
                            node_feat_every_type=node_feat_every_type,
                            edge_feat_every_type=edges_feat_every_type)

    """

    def __init__(self,
                 num_nodes_every_type,
                 edges_every_type,
                 node_feat_every_type=None,
                 edge_feat_every_type=None):

        self._num_nodes_dict = num_nodes_every_type
        self._edges_dict = edges_every_type
        if node_feat_every_type is not None:
            self._node_feat = node_feat_every_type
        else:
            self._node_feat = {}

        if edge_feat_every_type is not None:
            self._edge_feat = edge_feat_every_type
        else:
            self._edge_feat = {}

        self._multi_graph = {}
        for key, value in self._edges_dict.items():
            if not self._node_feat:
                node_feat = None
            else:
                node_feat = self._node_feat[key[0]]

            if not self._edge_feat:
                edge_feat = None
            else:
                edge_feat = self._edge_feat[key]

            self._multi_graph[key] = graph.Graph(
                num_nodes=self._num_nodes_dict[key[1]],
                edges=value,
                node_feat=node_feat,
                edge_feat=edge_feat)

    def __getitem__(self, edge_type):
        """__getitem__
        """
        return self._multi_graph[edge_type]

    def meta_path_random_walk(self, start_node, edge_types, meta_path,
                              max_depth):
        """Meta path random walk sampling.

        Args:
            start_nodes: int, node to begin random walk.
            edge_types: list, the edge types to be sampled.
            meta_path: 'user-item-user'
            max_depth: the max length of every walk.
        """
        edges_type_list = []
        node_type_list = meta_path.split('-')
        for i in range(1, len(node_type_list)):
            edges_type_list.append(
                (node_type_list[i - 1], node_type_list[i], edge_types[i - 1]))

        no_neighbors_flag = False
        walk = [start_node]
        for i in range(max_depth):
            for e_type in edges_type_list:
                cur_node = [walk[-1]]
                nxt_node = self._multi_graph[e_type].sample_successor(
                    cur_node, max_degree=1)  # list of np.array
                nxt_node = nxt_node[0]
                if len(nxt_node) == 0:
                    no_neighbors_flag = True
                    break
                else:
                    walk.append(nxt_node.tolist()[0])

            if no_neighbors_flag:
                break

        return walk

    def node_feat_info(self):
        """Return the information of node feature for HeterGraphWrapper.

        This function return the information of node features of all node types. And this
        function is used to help constructing HeterGraphWrapper

        Return:
            A dict of list of tuple (name, shape, dtype) for all given node feature.

        """
        node_feat_info = {}
        for node_type_name, feat_dict in self._node_feat.items():
            tmp_node_feat_info = []
            for feat_name, feat in feat_dict.items():
                full_name = feat_name
                tmp_node_feat_info.append(
                    (full_name, _hide_num_nodes(feat.shape), feat.dtype))
            node_feat_info[node_type_name] = tmp_node_feat_info

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
            A list of tuple ('srctype','dsttype', 'edges_type') for all edge types.
        
        """
        edge_types_info = []
        for key, _ in self._edges_dict.items():
            edge_types_info.append(key)

        return edge_types_info
