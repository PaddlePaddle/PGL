# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import json
import paddle
import copy
import numpy as np
import pickle as pkl
from collections import defaultdict

from pgl.graph import Graph
from pgl.utils import op
import pgl.graph_kernel as graph_kernel
from pgl.message import Message

__all__ = ['HeterGraph']

class HeterGraph(object):
    """Implementation of heterogeneous graph structure in pgl

    This is a simple implementation of heterogeneous graph structure in pgl.

    `pgl.HeterGraph` is an alias for `pgl.heter_graph.HeterGraph` 

    Args:

        edges: dict, every element in the dict is a list of (u, v) tuples or a 2D paddle.Tensor.

        num_nodes (optional): int, number of nodes in a heterogeneous graph

        node_types (optional): list of (u, node_type) tuples to specify the node type of every node
        node_feat (optional): a dict of numpy array as node features

        edge_feat (optional): a dict of dict as edge features for every edge type

    Examples:
        .. code-block:: python

            import numpy as np
            import pgl

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

            g = pgl.HeterGraph(
                            num_nodes=num_nodes,
                            edges=edges,
                            node_types=node_types,
                            node_feat=node_feat,
                            edge_feat=edges_feat)
    """

    def __init__(self,
                 edges,
                 num_nodes=None,
                 node_types=None,
                 node_feat=None,
                 edge_feat=None,
                 **kwargs):

        self._edges_dict = edges

        if isinstance(node_types, list):
            self._node_types = np.array(node_types, dtype=object)[:, 1]
        else:
            self._node_types = node_types

        if num_nodes is None:
            self._num_nodes = len(node_types)
        else:
            self._num_nodes = num_nodes

        self._nodes_type_dict = {}
        for ntype in np.unique(self._node_types):
            self._nodes_type_dict[ntype] = np.where(self._node_types == ntype)[0]

        if node_feat is not None:
            self._node_feat = node_feat
        else:
            self._node_feat = {}

        if edge_feat is not None:
            self._edge_feat = edge_feat
        else:
            self._edge_feat = {}

        if "multi_graph" in kwargs.keys():
            self._multi_graph = kwargs["multi_graph"]
        else:
            self._multi_graph = {}
            for etype, _edges in self._edges_dict.items():
                if not self._edge_feat:
                    edge_feat = None
                else:
                    edge_feat = self._edge_feat[etype]

                self._multi_graph[etype] = Graph(edges=_edges,
                        num_nodes=self._num_nodes,
                        node_feat=copy.deepcopy(self._node_feat),
                        edge_feat=edge_feat)

        self._edge_types = self.edge_types_info()
        self._nodes = None

        for etype, g in self._multi_graph.items():
            if g.is_tensor():
                self._is_tensor = True
            else:
                self._is_tensor = False
            break

    def is_tensor(self):
        """Return whether the HeterGraph is in paddle.Tensor.
        """
        return self._is_tensor

    @property
    def edge_types(self):
        """Return a list of edge types.
        """
        return self._edge_types

    @property
    def num_nodes(self):
        """Return the number of nodes.
        """
        _num_nodes = self._multi_graph[self._edge_types[0]].num_nodes
        return _num_nodes

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
    def edge_feat(self):
        """Return edge features of all edge types.
        """
        efeat = {}
        for etype, g in self._multi_graph.items():
            efeat[etype] = g.edge_feat
        return efeat

    @property
    def node_feat(self):
        """Return a dictionary of node features.
        """
        nfeat = self._multi_graph[self._edge_types[0]].node_feat
        return nfeat

    @property
    def nodes(self):
        """Return all nodes id from 0 to :code:`num_nodes - 1`
        """
        if self._nodes is None:
            if self.is_tensor():
                self._nodes = paddle.arange(self.num_nodes)
            else:
                self._nodes = np.arange(self.num_nodes)
        return self._nodes

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
            A numpy.ndarray or paddle.Tensor as the given nodes' indegree.
        """
        if edge_type is None:
            indegrees = []
            for e_type in self._edge_types:
                indegrees.append(self._multi_graph[e_type].indegree(nodes))
            if self.is_tensor():
                indegrees = paddle.sum(paddle.stack(indegrees), axis=0)
            else:
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
            A numpy.array or paddle.Tensor as the given nodes' outdegree.
        """
        if edge_type is None:
            outdegrees = []
            for e_type in self._edge_types:
                outdegrees.append(self._multi_graph[e_type].outdegree(nodes))
            if self.is_tensor():
                outdegrees = paddle.sum(paddle.stack(outdegrees), axis=0)
            else:
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

    def node_batch_iter(self, batch_size, shuffle=False, n_type=None):
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

        if self.is_tensor():
            nodes = paddle.to_tensor(nodes)
        start = 0
        while start < len(nodes):
            yield nodes[start:start + batch_size]
            start += batch_size

    def edge_types_info(self):
        """Return the information of all edge types.
        
        Return:
            A list of all edge types.
        
        """
        edge_types_info = []
        for key, _ in self._multi_graph.items():
            edge_types_info.append(key)

        return edge_types_info

    def tensor(self, inplace=True):
        """Convert the Heterogeneous Graph into paddle.Tensor format.

        In paddle.Tensor format, the graph edges and node features are in paddle.Tensor format.
        You can use send and recv in paddle.Tensor graph.
        
        Args:

            inplace: (Default True) Whether to convert the graph into tensor inplace. 

        """
        if self._is_tensor:
            return self

        if inplace:
            for etype in self._edge_types:
                self._multi_graph[etype].tensor(inplace)

            self._is_tensor = True
            return self
        else:
            new_multi_graph = {}
            for etype in self._edge_types:
                new_multi_graph[etype] = self._multi_graph[etype].tensor(inplace)

            new_graph = self.__class__(
                    edges=None,
                    node_types=self.__dict__["_node_types"],
                    multi_graph=new_multi_graph,
                    )
            return new_graph

    def numpy(self, inplace=True):
        """Convert the Heterogeneous Graph into numpy format.

        In numpy format, the graph edges and node features are in numpy.ndarray format.
        But you can't use send and recv in numpy graph.
        
        Args:

            inplace: (Default True) Whether to convert the graph into numpy inplace. 

        """
        if not self._is_tensor:
            return self

        if inplace:
            for etype in self._edge_types:
                self._multi_graph[etype].numpy(inplace)
            self._is_tensor = False
            return self
        else:
            new_multi_graph = {}
            for etype in self._edge_types:
                new_multi_graph[etype] = self._multi_graph[etype].numpy(inplace)

            new_graph = self.__class__(
                    edges=None,
                    node_types=self.__dict__["_node_types"],
                    multi_graph=new_multi_graph,
                    )
            return new_graph

    def dump(self, path, indegree=False, outdegree=False):
        """Dump the heterogeneous graph into a directory.

        This function will dump the graph information into the given directory path. 
        The graph can be read back with :code:`pgl.HeterGraph.load`

        Args:
            path: The directory for the storage of the heterogeneous graph.

        """
        if indegree:
            for etype, g in self._multi_graph.items():
                g.indegree()

        if outdegree:
            for etype, g in self._multi_graph.items():
                g.outdegree()

        if not os.path.exists(path):
            os.makedirs(path)

        np.save(os.path.join(path, "node_types.npy"), self._node_types)
        with open(os.path.join(path, "edge_types.pkl"), "wb") as f:
            pkl.dump(self._edge_types, f)

        for etype, g in self._multi_graph.items():
            sub_path = os.path.join(path, etype)
            g.dump(sub_path)

    @classmethod
    def load(cls, path, mmap_mode="r"):
        """Load HeterGraph from path and return a HeterGraph instance in numpy. 

        Args:

            path: The directory path of the stored HeterGraph.

            mmap_mode: Default :code:`mmap_mode="r"`. If not None, memory-map the graph.  
        """

        _node_types = np.load(os.path.join(path, "node_types.npy"), allow_pickle=True)

        with open(os.path.join(path, "edge_types.pkl"), "rb") as f:
            _edge_types = pkl.load(f)

        _multi_graph = {}

        for etype in _edge_types:
            sub_path = os.path.join(path, etype)
            _multi_graph[etype] = Graph.load(sub_path, mmap_mode)

        return cls(edges=None,
                node_types=_node_types,
                multi_graph=_multi_graph,
                )




