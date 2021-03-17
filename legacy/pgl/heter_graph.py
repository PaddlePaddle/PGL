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
import os
import time
import numpy as np
import pickle as pkl
import time
import pgl.graph_kernel as graph_kernel
from pgl.graph import Graph, MemmapGraph

__all__ = ['HeterGraph', 'SubHeterGraph']


def _hide_num_nodes(shape):
    """Set the first dimension as unknown
    """
    shape = list(shape)
    shape[0] = None
    return shape


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

        if isinstance(node_types, list):
            self._node_types = np.array(node_types, dtype=object)[:, 1]
        else:
            self._node_types = node_types

        self._nodes_type_dict = {}
        for n_type in np.unique(self._node_types):
            self._nodes_type_dict[n_type] = np.where(
                self._node_types == n_type)[0]

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

            self._multi_graph[key] = Graph(
                num_nodes=self._num_nodes,
                edges=value,
                node_feat=self._node_feat,
                edge_feat=edge_feat)

        self._edge_types = self.edge_types_info()

    def dump(self, path, indegree=False, outdegree=False):

        if indegree:
            for e_type, g in self._multi_graph.items():
                g.indegree()

        if outdegree:
            for e_type, g in self._multi_graph.items():
                g.outdegree()

        if not os.path.exists(path):
            os.makedirs(path)

        np.save(os.path.join(path, "num_nodes.npy"), self._num_nodes)
        np.save(os.path.join(path, "node_types.npy"), self._node_types)
        with open(os.path.join(path, "edge_types.pkl"), 'wb') as f:
            pkl.dump(self._edge_types, f)
        with open(os.path.join(path, "nodes_type_dict.pkl"), 'wb') as f:
            pkl.dump(self._nodes_type_dict, f)

        for e_type, g in self._multi_graph.items():
            sub_path = os.path.join(path, e_type)
            g.dump(sub_path)

    @property
    def edge_types(self):
        """Return a list of edge types.
        """
        return self._edge_types

    @property
    def num_nodes(self):
        """Return the number of nodes.
        """
        return self._num_nodes

    @property
    def num_edges(self):
        """Return edges number of all edge types.
        """
        n_edges = {}
        for e_type in self._edge_types:
            n_edges[e_type] = self._multi_graph[e_type].num_edges
        return n_edges

    @property
    def node_types(self):
        """Return the node types.
        """
        return self._node_types

    @property
    def edge_feat(self, edge_type=None):
        """Return edge features of all edge types.
        """
        return self._edge_feat

    @property
    def node_feat(self):
        """Return a dictionary of node features.
        """
        return self._node_feat

    @property
    def nodes(self):
        """Return all nodes id from 0 to :code:`num_nodes - 1`
        """
        return np.arange(self._num_nodes, dtype='int64')

    def __getitem__(self, edge_type):
        """__getitem__
        """
        return self._multi_graph[edge_type]

    def num_nodes_by_type(self, n_type=None):
        """Return the number of nodes with the specified node type.
        """
        if n_type not in self._nodes_type_dict:
            raise ("%s is not in valid node type" % n_type)
        else:
            return len(self._nodes_type_dict[n_type])

    def indegree(self, nodes=None, edge_type=None):
        """Return the indegree of the given nodes with the specified edge_type.

        Args:
            nodes: Return the indegree of given nodes.
                    if nodes is None, return indegree for all nodes.

            edge_types: Return the indegree with specified edge_type.
                    if edge_type is None, return the total indegree of the given nodes.

        Return:
            A numpy.ndarray as the given nodes' indegree.
        """
        if edge_type is None:
            indegrees = []
            for e_type in self._edge_types:
                indegrees.append(self._multi_graph[e_type].indegree(nodes))
            indegrees = np.sum(np.vstack(indegrees), axis=0)
            return indegrees
        else:
            return self._multi_graph[edge_type].indegree(nodes)

    def outdegree(self, nodes=None, edge_type=None):
        """Return the outdegree of the given nodes with the specified edge_type.

        Args:
            nodes: Return the outdegree of given nodes,
                   if nodes is None, return outdegree for all nodes

            edge_types: Return the outdegree with specified edge_type.
                    if edge_type is None, return the total outdegree of the given nodes.

        Return:
            A numpy.array as the given nodes' outdegree.
        """
        if edge_type is None:
            outdegrees = []
            for e_type in self._edge_types:
                outdegrees.append(self._multi_graph[e_type].outdegree(nodes))
            outdegrees = np.sum(np.vstack(outdegrees), axis=0)
            return outdegrees
        else:
            return self._multi_graph[edge_type].outdegree(nodes)

    def successor(self, edge_type, nodes=None, return_eids=False):
        """Find successor of given nodes with the specified edge_type.

        Args:
            nodes: Return the successor of given nodes,
                   if nodes is None, return successor for all nodes

            edge_types: Return the successor with specified edge_type.
                    if edge_type is None, return the total successor of the given nodes
                    and eids are invalid in this way.

            return_eids: If True return nodes together with corresponding eid
        """
        return self._multi_graph[edge_type].successor(nodes, return_eids)

    def sample_successor(self,
                         edge_type,
                         nodes,
                         max_degree,
                         return_eids=False,
                         shuffle=False):
        """Sample successors of given nodes with the specified edge_type.

        Args:
            edge_type: The specified edge_type.

            nodes: Given nodes whose successors will be sampled.

            max_degree: The max sampled successors for each nodes.

            return_eids: Whether to return the corresponding eids.

        Return:

            Return a list of numpy.ndarray and each numpy.ndarray represent a list
            of sampled successor ids for given nodes with specified edge type. 
            If :code:`return_eids=True`, there will be an additional list of 
            numpy.ndarray and each numpy.ndarray represent a list of eids that 
            connected nodes to their successors.
        """
        return self._multi_graph[edge_type].sample_successor(
            nodes=nodes,
            max_degree=max_degree,
            return_eids=return_eids,
            shuffle=shuffle)

    def predecessor(self, edge_type, nodes=None, return_eids=False):
        """Find predecessor of given nodes with the specified edge_type.

        Args:
            nodes: Return the predecessor of given nodes,
                   if nodes is None, return predecessor for all nodes

            edge_types: Return the predecessor with specified edge_type.

            return_eids: If True return nodes together with corresponding eid
        """
        return self._multi_graph[edge_type].predecessor(nodes, return_eids)

    def sample_predecessor(self,
                           edge_type,
                           nodes,
                           max_degree,
                           return_eids=False,
                           shuffle=False):
        """Sample predecessors of given nodes with the specified edge_type.

        Args:
            edge_type: The specified edge_type.

            nodes: Given nodes whose predecessors will be sampled.

            max_degree: The max sampled predecessors for each nodes.

            return_eids: Whether to return the corresponding eids.

        Return:

            Return a list of numpy.ndarray and each numpy.ndarray represent a list
            of sampled predecessor ids for given nodes with specified edge type. 
            If :code:`return_eids=True`, there will be an additional list of 
            numpy.ndarray and each numpy.ndarray represent a list of eids that 
            connected nodes to their predecessors.
        """
        return self._multi_graph[edge_type].sample_predecessor(
            nodes=nodes,
            max_degree=max_degree,
            return_eids=return_eids,
            shuffle=shuffle)

    def node_batch_iter(self, batch_size, shuffle=True, n_type=None):
        """Node batch iterator

        Iterate all nodes by batch with the specified node type.

        Args:
            batch_size: The batch size of each batch of nodes.

            shuffle: Whether shuffle the nodes.
            
            n_type: Iterate the nodes with the specified node type. If n_type is None, 
                    iterate all nodes by batch.

        Return:
            Batch iterator
        """
        if n_type is None:
            nodes = np.arange(self._num_nodes, dtype="int64")
        else:
            nodes = self._nodes_type_dict[n_type]

        if shuffle:
            np.random.shuffle(nodes)
        start = 0
        while start < len(nodes):
            yield nodes[start:start + batch_size]
            start += batch_size

    def sample_nodes(self, sample_num, n_type=None):
        """Sample nodes with the specified n_type from the graph

        This function helps to sample nodes with the specified n_type from the graph.
        If n_type is None, this function will sample nodes from all nodes.
        Nodes might be duplicated.

        Args:
            sample_num: The number of samples
            n_type: The nodes of type to be sampled

        Return:
            A list of nodes
        """
        if n_type is not None:
            return np.random.choice(
                self._nodes_type_dict[n_type], size=sample_num)
        else:
            return np.random.randint(
                low=0, high=self._num_nodes, size=sample_num)

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
        for key, _ in self._multi_graph.items():
            edge_types_info.append(key)

        return edge_types_info


class SubHeterGraph(HeterGraph):
    """Implementation of SubHeterGraph in pgl.

    SubHeterGraph is inherit from :code:`HeterGraph`. 

    Args:
        num_nodes: number of nodes in a heterogeneous graph
        edges: dict, every element in dict is a list of (u, v) tuples.
        node_types (optional): list of (u, node_type) tuples to specify the node type of every node
        node_feat (optional): a dict of numpy array as node features
        edge_feat (optional): a dict of dict as edge features for every edge type

        reindex: A dictionary that maps parent hetergraph node id to subhetergraph node id.
    """

    def __init__(self,
                 num_nodes,
                 edges,
                 node_types=None,
                 node_feat=None,
                 edge_feat=None,
                 reindex=None):
        super(SubHeterGraph, self).__init__(
            num_nodes=num_nodes,
            edges=edges,
            node_types=node_types,
            node_feat=node_feat,
            edge_feat=edge_feat)

        if reindex is None:
            reindex = {}
        self._from_reindex = reindex
        self._to_reindex = {u: v for v, u in reindex.items()}

    def reindex_from_parrent_nodes(self, nodes):
        """Map the given parent graph node id to subgraph id.

        Args:
            nodes: A list of nodes from parent graph.

        Return:
            A list of subgraph ids.
        """
        return graph_kernel.map_nodes(nodes, self._from_reindex)

    def reindex_to_parrent_nodes(self, nodes):
        """Map the given subgraph node id to parent graph id.

        Args:
            nodes: A list of nodes in this subgraph.

        Return:
            A list of node ids in parent graph.
        """
        return graph_kernel.map_nodes(nodes, self._to_reindex)


class MemmapHeterGraph(HeterGraph):
    def __init__(self, path):
        self._num_nodes = np.load(os.path.join(path, 'num_nodes.npy'))
        self._node_types = np.load(
            os.path.join(path, 'node_types.npy'), allow_pickle=True)

        with open(os.path.join(path, 'edge_types.pkl'), 'rb') as f:
            self._edge_types = pkl.load(f)

        with open(os.path.join(path, "nodes_type_dict.pkl"), 'rb') as f:
            self._nodes_type_dict = pkl.load(f)

        self._multi_graph = {}
        for e_type in self._edge_types:
            sub_path = os.path.join(path, e_type)
            self._multi_graph[e_type] = MemmapGraph(sub_path)
