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


def is_all(arg):
    """is_all
    """
    return isinstance(arg, str) and arg == ALL


class BipartiteGraphWrapper(GraphWrapper):
    """Implement a bipartite graph wrapper that creates a graph data holders.
    """

    def __init__(self, name, place, node_feat=[], edge_feat=[]):
        super(BipartiteGraphWrapper, self).__init__(name, place, node_feat,
                                                    edge_feat)

    def send(self,
             message_func,
             src_nfeat_list=None,
             dst_nfeat_list=None,
             efeat_list=None):
        """Send message from all src nodes to dst nodes.

        The UDF message function should has the following format.

        .. code-block:: python

            def message_func(src_feat, dst_feat, edge_feat):
                '''
                    Args:
                        src_feat: the node feat dict attached to the src nodes.
                        dst_feat: the node feat dict attached to the dst nodes.
                        edge_feat: the edge feat dict attached to the
                                   corresponding (src, dst) edges.

                    Return:
                        It should return a tensor or a dictionary of tensor. And each tensor
                        should have a shape of (num_edges, dims).
                '''
                pass

        Args:
            message_func: UDF function.
            src_nfeat_list: a list of tuple (name, tensor) for src nodes
            dst_nfeat_list: a list of tuple (name, tensor) for dst nodes
            efeat_list: a list of names or tuple (name, tensor)

        Return:
            A dictionary of tensor representing the message. Each of the values
            in the dictionary has a shape (num_edges, dim) which should be collected
            by :code:`recv` function.
        """
        if efeat_list is None:
            efeat_list = {}
        if src_nfeat_list is None:
            src_nfeat_list = {}
        if dst_nfeat_list is None:
            dst_nfeat_list = {}

        src, dst = self.edges
        src_feat = {}
        for feat in src_nfeat_list:
            if isinstance(feat, str):
                src_feat[feat] = self.node_feat[feat]
            else:
                name, tensor = feat
                src_feat[name] = tensor

        dst_feat = {}
        for feat in dst_nfeat_list:
            if isinstance(feat, str):
                dst_feat[feat] = self.node_feat[feat]
            else:
                name, tensor = feat
                dst_feat[name] = tensor

        efeat = {}
        for feat in efeat_list:
            if isinstance(feat, str):
                efeat[feat] = self.edge_feat[feat]
            else:
                name, tensor = feat
                efeat[name] = tensor

        src_feat = op.read_rows(src_feat, src)
        dst_feat = op.read_rows(dst_feat, dst)
        msg = message_func(src_feat, dst_feat, efeat)

        return msg


class HeterGraphWrapper(object):
    """Implement a heterogeneous graph wrapper that creates a graph data holders
    that attributes and features in the heterogeneous graph.
    And we provide interface :code:`to_feed` to help converting :code:`Graph`
    data into :code:`feed_dict`.

    Args:
        name: The heterogeneous graph data prefix

        place: fluid.CPUPlace or fluid.GPUPlace(n) indicating the
               device to hold the graph data.

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


                place = fluid.CPUPlace()

                gw = pgl.heter_graph_wrapper.HeterGraphWrapper(
                                    name='heter_graph', 
                                    place = place, 
                                    edge_types = g.edge_types_info(),
                                    node_feat=g.node_feat_info(),
                                    edge_feat=g.edge_feat_info())
    """

    def __init__(self, name, place, edge_types, node_feat={}, edge_feat={}):
        self.__data_name_prefix = name
        self._place = place
        self._edge_types = edge_types
        self._multi_gw = {}
        for edge_type in self._edge_types:
            type_name = self.__data_name_prefix + '/' + edge_type[
                0] + '_' + edge_type[1]
            if node_feat:
                n_feat = node_feat[edge_type[0]]
            else:
                n_feat = {}

            if edge_feat:
                e_feat = edge_feat[edge_type]
            else:
                e_feat = {}

            self._multi_gw[edge_type] = BipartiteGraphWrapper(
                name=type_name,
                place=self._place,
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
