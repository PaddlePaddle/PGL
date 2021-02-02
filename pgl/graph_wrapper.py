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
import paddle.fluid.layers as L

from pgl.utils import op
from pgl.utils import paddle_helper
from pgl.utils.logger import log

__all__ = [
    "BaseGraphWrapper", "GraphWrapper", "StaticGraphWrapper",
    "BatchGraphWrapper"
]


def send(src, dst, nfeat, efeat, message_func):
    """Send message from src to dst.
    """
    src_feat = op.RowReader(nfeat, src)
    dst_feat = op.RowReader(nfeat, dst)
    msg = message_func(src_feat, dst_feat, efeat)
    return msg


def recv(dst, uniq_dst, bucketing_index, msg, reduce_function, num_nodes,
         num_edges):
    """Recv message from given msg to dst nodes.
    """
    if reduce_function == "sum":
        if isinstance(msg, dict):
            raise TypeError("The message for build-in function"
                            " should be Tensor not dict.")

        try:
            out_dim = msg.shape[-1]
            init_output = L.fill_constant(
                shape=[num_nodes, out_dim], value=0, dtype=msg.dtype)
            init_output.stop_gradient = False
            empty_msg_flag = L.cast(num_edges > 0, dtype=msg.dtype)
            msg = msg * empty_msg_flag
            output = paddle_helper.scatter_add(init_output, dst, msg)
            return output
        except TypeError as e:
            warnings.warn(
                "scatter_add is not supported with paddle version <= 1.5")

            def sum_func(message):
                return L.sequence_pool(message, "sum")

            reduce_function = sum_func

    bucketed_msg = op.nested_lod_reset(msg, bucketing_index)
    output = reduce_function(bucketed_msg)
    output_dim = output.shape[-1]

    empty_msg_flag = L.cast(num_edges > 0, dtype=output.dtype)
    output = output * empty_msg_flag

    init_output = L.fill_constant(
        shape=[num_nodes, output_dim], value=0, dtype=output.dtype)
    init_output.stop_gradient = True
    final_output = L.scatter(init_output, uniq_dst, output)
    return final_output


class BaseGraphWrapper(object):
    """This module implement base class for graph wrapper.

    Currently our PGL is developed based on static computational mode of
    paddle (we'll support dynamic computational model later). We need to build
    model upon a virtual data holder. BaseGraphWrapper provide a virtual
    graph structure that users can build deep learning models
    based on this virtual graph. And then feed real graph data to run
    the models. Moreover, we provide convenient message-passing interface
    (send & recv) for building graph neural networks.

    NOTICE: Don't use this BaseGraphWrapper directly. Use :code:`GraphWrapper`
    and :code:`StaticGraphWrapper` to create graph wrapper instead.
    """

    def __init__(self):
        self.node_feat_tensor_dict = {}
        self.edge_feat_tensor_dict = {}
        self._edges_src = None
        self._edges_dst = None
        self._num_nodes = None
        self._indegree = None
        self._edge_uniq_dst = None
        self._edge_uniq_dst_count = None
        self._graph_lod = None
        self._num_graph = None
        self._num_edges = None
        self._data_name_prefix = ""

    def __repr__(self):
        return self._data_name_prefix

    def send(self, message_func, nfeat_list=None, efeat_list=None):
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
            nfeat_list: a list of names or tuple (name, tensor)
            efeat_list: a list of names or tuple (name, tensor)

        Return:
            A dictionary of tensor representing the message. Each of the values
            in the dictionary has a shape (num_edges, dim) which should be collected
            by :code:`recv` function.
        """
        if efeat_list is None:
            efeat_list = {}
        else:
            warnings.warn(
                "The edge features in argument `efeat_list` should be fetched "
                "from a instance of `pgl.graph_wrapper.GraphWrapper`, "
                "because we have sorted the edges and the order of edges is changed.\n"
                "Therefore, if you use external edge features, "
                "the order of features of each edge may not match its edge, "
                "which can cause serious errors.\n"
                "If you use the `efeat_list` correctly, please ignore this warning."
            )

        if nfeat_list is None:
            nfeat_list = {}

        src, dst = self.edges
        nfeat = {}

        for feat in nfeat_list:
            if isinstance(feat, str):
                nfeat[feat] = self.node_feat[feat]
            else:
                name, tensor = feat
                nfeat[name] = tensor

        efeat = {}
        for feat in efeat_list:
            if isinstance(feat, str):
                efeat[feat] = self.edge_feat[feat]
            else:
                name, tensor = feat
                efeat[name] = tensor

        msg = send(src, dst, nfeat, efeat, message_func)
        return msg

    def recv(self, msg, reduce_function):
        """Recv message and aggregate the message by reduce_fucntion

        The UDF reduce_function function should has the following format.

        .. code-block:: python

            def reduce_func(msg):
                '''
                    Args:
                        msg: A LodTensor or a dictionary of LodTensor whose batch_size
                             is equals to the number of unique dst nodes.

                    Return:
                        It should return a tensor with shape (batch_size, out_dims). The
                        batch size should be the same as msg.
                '''
                pass

        Args:
            msg: A tensor or a dictionary of tensor created by send function..

            reduce_function: UDF reduce function or strings "sum" as built-in function.
                             The built-in "sum" will use scatter_add to optimized the speed.

        Return:
            A tensor with shape (num_nodes, out_dims). The output for nodes with no message
            will be zeros.
        """
        output = recv(
            dst=self._edges_dst,
            uniq_dst=self._edge_uniq_dst,
            bucketing_index=self._edge_uniq_dst_count,
            msg=msg,
            reduce_function=reduce_function,
            num_edges=self._num_edges,
            num_nodes=self._num_nodes)
        return output

    @property
    def edges(self):
        """Return a tuple of edge Tensor (src, dst).

        Return:
            A tuple of Tensor (src, dst). Src and dst are both
            tensor with shape (num_edges, ) and dtype int64.
        """
        return self._edges_src, self._edges_dst

    @property
    def num_nodes(self):
        """Return a variable of number of nodes

        Return:
            A variable with shape (1,) as the number of nodes in int64.
        """
        return self._num_nodes

    @property
    def graph_lod(self):
        """Return graph index for graphs

        Return:
            A variable with shape [None ]  as the Lod information of multiple-graph.
        """
        return self._graph_lod

    @property
    def num_graph(self):
        """Return a variable of number of graphs

        Return:
            A variable with shape (1,) as the number of Graphs in int64.
        """
        return self._num_graph

    @property
    def edge_feat(self):
        """Return a dictionary of tensor representing edge features.

        Return:
            A dictionary whose keys are the feature names and the values
            are feature tensor.
        """
        return self.edge_feat_tensor_dict

    @property
    def node_feat(self):
        """Return a dictionary of tensor representing node features.

        Return:
            A dictionary whose keys are the feature names and the values
            are feature tensor.
        """
        return self.node_feat_tensor_dict

    def indegree(self):
        """Return the indegree tensor for all nodes.

        Return:
            A tensor of shape (num_nodes, ) in int64.
        """
        return self._indegree


class StaticGraphWrapper(BaseGraphWrapper):
    """Implement a graph wrapper that the data of the graph won't
    be changed and it can be fit into the GPU or CPU memory. This
    can reduce the time of swapping large data from GPU and CPU.

    Args:
        name: The graph data prefix

        graph: The static graph that should be put into memory

        place: fluid.CPUPlace or fluid.CUDAPlace(n) indicating the
               device to hold the graph data.

    Examples:

        If we have a immutable graph and it can be fit into the GPU or CPU.
        we can just use a :code:`StaticGraphWrapper` to pre-place the graph
        data into devices.

        .. code-block:: python

            import numpy as np
            import paddle.fluid as fluid
            from pgl.graph import Graph
            from pgl.graph_wrapper import StaticGraphWrapper

            place = fluid.CPUPlace()
            exe = fluid.Excecutor(place)

            num_nodes = 5
            edges = [ (0, 1), (1, 2), (3, 4)]
            feature = np.random.randn(5, 100)
            edge_feature = np.random.randn(3, 100)
            graph = Graph(num_nodes=num_nodes,
                        edges=edges,
                        node_feat={
                            "feature": feature
                        },
                        edge_feat={
                            "edge_feature": edge_feature
                        })

            graph_wrapper = StaticGraphWrapper(name="graph",
                        graph=graph,
                        place=place)

            # build your deep graph model

            # Initialize parameters for deep graph model
            exe.run(fluid.default_startup_program())

            # Initialize graph data
            graph_wrapper.initialize(place)
    """

    def __init__(self, name, graph, place):
        super(StaticGraphWrapper, self).__init__()
        self._data_name_prefix = name
        self._initializers = []
        self.__create_graph_attr(graph)

    def __create_graph_attr(self, graph):
        """Create graph attributes for paddlepaddle.
        """
        src, dst, eid = graph.sorted_edges(sort_by="dst")
        indegree = graph.indegree()
        nodes = graph.nodes
        uniq_dst = nodes[indegree > 0]
        uniq_dst_count = indegree[indegree > 0]
        uniq_dst_count = np.cumsum(uniq_dst_count, dtype='int32')
        uniq_dst_count = np.insert(uniq_dst_count, 0, 0)
        graph_lod = graph.graph_lod
        num_graph = graph.num_graph

        num_edges = len(src)
        if num_edges == 0:
            # Fake Graph
            src = np.array([0], dtype="int64")
            dst = np.array([0], dtype="int64")
            eid = np.array([0], dtype="int64")
            uniq_dst_count = np.array([0, 1], dtype="int32")
            uniq_dst = np.array([0], dtype="int64")

        edge_feat = {}

        for key, value in graph.edge_feat.items():
            edge_feat[key] = value[eid]
        node_feat = graph.node_feat

        self.__create_graph_node_feat(node_feat, self._initializers)
        self.__create_graph_edge_feat(edge_feat, self._initializers)

        self._num_edges, init = paddle_helper.constant(
            dtype="int64",
            value=np.array(
                [num_edges], dtype="int64"),
            name=self._data_name_prefix + '/num_edges')
        self._initializers.append(init)

        self._num_graph, init = paddle_helper.constant(
            dtype="int64",
            value=np.array(
                [num_graph], dtype="int64"),
            name=self._data_name_prefix + '/num_graph')
        self._initializers.append(init)

        self._edges_src, init = paddle_helper.constant(
            dtype="int64",
            value=src,
            name=self._data_name_prefix + '/edges_src')
        self._initializers.append(init)

        self._edges_dst, init = paddle_helper.constant(
            dtype="int64",
            value=dst,
            name=self._data_name_prefix + '/edges_dst')
        self._initializers.append(init)

        self._num_nodes, init = paddle_helper.constant(
            dtype="int64",
            hide_batch_size=False,
            value=np.array([graph.num_nodes]),
            name=self._data_name_prefix + '/num_nodes')
        self._initializers.append(init)

        self._edge_uniq_dst, init = paddle_helper.constant(
            name=self._data_name_prefix + "/uniq_dst",
            dtype="int64",
            value=uniq_dst)
        self._initializers.append(init)

        self._edge_uniq_dst_count, init = paddle_helper.constant(
            name=self._data_name_prefix + "/uniq_dst_count",
            dtype="int32",
            value=uniq_dst_count)
        self._initializers.append(init)

        self._graph_lod, init = paddle_helper.constant(
            name=self._data_name_prefix + "/graph_lod",
            dtype="int32",
            value=graph_lod)
        self._initializers.append(init)

        self._indegree, init = paddle_helper.constant(
            name=self._data_name_prefix + "/indegree",
            dtype="int64",
            value=indegree)
        self._initializers.append(init)

    def __create_graph_node_feat(self, node_feat, collector):
        """Convert node features into paddlepaddle tensor.
        """
        for node_feat_name, node_feat_value in node_feat.items():
            node_feat_shape = node_feat_value.shape
            node_feat_dtype = node_feat_value.dtype
            self.node_feat_tensor_dict[
                node_feat_name], init = paddle_helper.constant(
                    name=self._data_name_prefix + '/node_feat/' +
                    node_feat_name,
                    dtype=node_feat_dtype,
                    value=node_feat_value)
            collector.append(init)

    def __create_graph_edge_feat(self, edge_feat, collector):
        """Convert edge features into paddlepaddle tensor.
        """
        for edge_feat_name, edge_feat_value in edge_feat.items():
            edge_feat_shape = edge_feat_value.shape
            edge_feat_dtype = edge_feat_value.dtype
            self.edge_feat_tensor_dict[
                edge_feat_name], init = paddle_helper.constant(
                    name=self._data_name_prefix + '/edge_feat/' +
                    edge_feat_name,
                    dtype=edge_feat_dtype,
                    value=edge_feat_value)
            collector.append(init)

    def initialize(self, place):
        """Placing the graph data into the devices.

        Args:
            place: fluid.CPUPlace or fluid.CUDAPlace(n) indicating the
                   device to hold the graph data.
        """
        log.info(
            "StaticGraphWrapper.initialize must be called after startup program"
        )
        for init_func in self._initializers:
            init_func(place)


class GraphWrapper(BaseGraphWrapper):
    """Implement a graph wrapper that creates a graph data holders
    that attributes and features in the graph are :code:`L.data`.
    And we provide interface :code:`to_feed` to help converting :code:`Graph`
    data into :code:`feed_dict`.

    Args:
        name: The graph data prefix

        node_feat: A list of tuples that decribe the details of node
                   feature tenosr. Each tuple mush be (name, shape, dtype)
                   and the first dimension of the shape must be set unknown
                   (-1 or None) or we can easily use :code:`Graph.node_feat_info()`
                   to get the node_feat settings.

        edge_feat: A list of tuples that decribe the details of edge
                   feature tenosr. Each tuple mush be (name, shape, dtype)
                   and the first dimension of the shape must be set unknown
                   (-1 or None) or we can easily use :code:`Graph.edge_feat_info()`
                   to get the edge_feat settings.

    Examples:

        .. code-block:: python

            import numpy as np
            import paddle.fluid as fluid
            from pgl.graph import Graph
            from pgl.graph_wrapper import GraphWrapper

            place = fluid.CPUPlace()
            exe = fluid.Excecutor(place)

            num_nodes = 5
            edges = [ (0, 1), (1, 2), (3, 4)]
            feature = np.random.randn(5, 100)
            edge_feature = np.random.randn(3, 100)
            graph = Graph(num_nodes=num_nodes,
                        edges=edges,
                        node_feat={
                            "feature": feature
                        },
                        edge_feat={
                            "edge_feature": edge_feature
                        })

            graph_wrapper = GraphWrapper(name="graph",
                        node_feat=graph.node_feat_info(),
                        edge_feat=graph.edge_feat_info())

            # build your deep graph model
            ...

            # Initialize parameters for deep graph model
            exe.run(fluid.default_startup_program())

            for i in range(10):
                feed_dict = graph_wrapper.to_feed(graph)
                ret = exe.run(fetch_list=[...], feed=feed_dict )
    """

    def __init__(self, name, node_feat=[], edge_feat=[], **kwargs):
        super(GraphWrapper, self).__init__()
        # collect holders for PyReader
        self._data_name_prefix = name
        self._holder_list = []
        self.__create_graph_attr_holders()
        for node_feat_name, node_feat_shape, node_feat_dtype in node_feat:
            self.__create_graph_node_feat_holders(
                node_feat_name, node_feat_shape, node_feat_dtype)

        for edge_feat_name, edge_feat_shape, edge_feat_dtype in edge_feat:
            self.__create_graph_edge_feat_holders(
                edge_feat_name, edge_feat_shape, edge_feat_dtype)

    def __create_graph_attr_holders(self):
        """Create data holders for graph attributes.
        """
        self._num_edges = L.data(
            self._data_name_prefix + '/num_edges',
            shape=[1],
            append_batch_size=False,
            dtype="int64",
            stop_gradient=True)
        self._num_graph = L.data(
            self._data_name_prefix + '/num_graph',
            shape=[1],
            append_batch_size=False,
            dtype="int64",
            stop_gradient=True)
        self._edges_src = L.data(
            self._data_name_prefix + '/edges_src',
            shape=[None],
            append_batch_size=False,
            dtype="int64",
            stop_gradient=True)
        self._edges_dst = L.data(
            self._data_name_prefix + '/edges_dst',
            shape=[None],
            append_batch_size=False,
            dtype="int64",
            stop_gradient=True)
        self._num_nodes = L.data(
            self._data_name_prefix + '/num_nodes',
            shape=[1],
            append_batch_size=False,
            dtype='int64',
            stop_gradient=True)

        self._edge_uniq_dst = L.data(
            self._data_name_prefix + "/uniq_dst",
            shape=[None],
            append_batch_size=False,
            dtype="int64",
            stop_gradient=True)

        self._graph_lod = L.data(
            self._data_name_prefix + "/graph_lod",
            shape=[None],
            append_batch_size=False,
            dtype="int32",
            stop_gradient=True)

        self._edge_uniq_dst_count = L.data(
            self._data_name_prefix + "/uniq_dst_count",
            shape=[None],
            append_batch_size=False,
            dtype="int32",
            stop_gradient=True)

        self._indegree = L.data(
            self._data_name_prefix + "/indegree",
            shape=[None],
            append_batch_size=False,
            dtype="int64",
            stop_gradient=True)
        self._holder_list.extend([
            self._edges_src,
            self._edges_dst,
            self._num_nodes,
            self._edge_uniq_dst,
            self._edge_uniq_dst_count,
            self._indegree,
            self._graph_lod,
            self._num_graph,
            self._num_edges,
        ])

    def __create_graph_node_feat_holders(self, node_feat_name, node_feat_shape,
                                         node_feat_dtype):
        """Create data holders for node features.
        """
        feat_holder = L.data(
            self._data_name_prefix + '/node_feat/' + node_feat_name,
            shape=node_feat_shape,
            append_batch_size=False,
            dtype=node_feat_dtype,
            stop_gradient=True)
        self.node_feat_tensor_dict[node_feat_name] = feat_holder
        self._holder_list.append(feat_holder)

    def __create_graph_edge_feat_holders(self, edge_feat_name, edge_feat_shape,
                                         edge_feat_dtype):
        """Create edge holders for edge features.
        """
        feat_holder = L.data(
            self._data_name_prefix + '/edge_feat/' + edge_feat_name,
            shape=edge_feat_shape,
            append_batch_size=False,
            dtype=edge_feat_dtype,
            stop_gradient=True)
        self.edge_feat_tensor_dict[edge_feat_name] = feat_holder
        self._holder_list.append(feat_holder)

    def to_feed(self, graph):
        """Convert the graph into feed_dict.

        This function helps to convert graph data into feed dict
        for :code:`fluid.Excecutor` to run the model.

        Args:
            graph: the :code:`Graph` data object

        Return:
            A dictionary contains data holder names and its corresponding
            data.
        """
        feed_dict = {}
        src, dst, eid = graph.sorted_edges(sort_by="dst")
        indegree = graph.indegree()
        nodes = graph.nodes
        num_edges = len(src)
        uniq_dst = nodes[indegree > 0]
        uniq_dst_count = indegree[indegree > 0]
        uniq_dst_count = np.cumsum(uniq_dst_count, dtype='int32')
        uniq_dst_count = np.insert(uniq_dst_count, 0, 0)
        num_graph = graph.num_graph
        graph_lod = graph.graph_lod

        if num_edges == 0:
            # Fake Graph
            src = np.array([0], dtype="int64")
            dst = np.array([0], dtype="int64")
            eid = np.array([0], dtype="int64")

            uniq_dst_count = np.array([0, 1], dtype="int32")
            uniq_dst = np.array([0], dtype="int64")

        edge_feat = {}

        for key, value in graph.edge_feat.items():
            edge_feat[key] = value[eid]
        node_feat = graph.node_feat

        feed_dict[self._data_name_prefix + '/num_edges'] = np.array(
            [num_edges], dtype="int64")
        feed_dict[self._data_name_prefix + '/edges_src'] = src
        feed_dict[self._data_name_prefix + '/edges_dst'] = dst
        feed_dict[self._data_name_prefix + '/num_nodes'] = np.array(
            [graph.num_nodes], dtype="int64")
        feed_dict[self._data_name_prefix + '/uniq_dst'] = uniq_dst
        feed_dict[self._data_name_prefix + '/uniq_dst_count'] = uniq_dst_count
        feed_dict[self._data_name_prefix + '/indegree'] = indegree
        feed_dict[self._data_name_prefix + '/graph_lod'] = graph_lod
        feed_dict[self._data_name_prefix + '/num_graph'] = np.array(
            [num_graph], dtype="int64")
        feed_dict[self._data_name_prefix + '/indegree'] = indegree

        for key in self.node_feat_tensor_dict:
            feed_dict[self._data_name_prefix + '/node_feat/' +
                      key] = node_feat[key]

        for key in self.edge_feat_tensor_dict:
            feed_dict[self._data_name_prefix + '/edge_feat/' +
                      key] = edge_feat[key]

        return feed_dict

    @property
    def holder_list(self):
        """Return the holder list.
        """
        return self._holder_list


def get_degree(edge, num_nodes):
    init_output = L.fill_constant(shape=[num_nodes], value=0, dtype="float32")
    init_output.stop_gradient = True
    final_output = L.scatter(
        init_output,
        edge,
        L.full_like(
            edge, 1, dtype="float32"),
        overwrite=False)
    return final_output


class DropEdgeWrapper(BaseGraphWrapper):
    """Implement of Edge Drop """

    def __init__(self, graph_wrapper, dropout, keep_self_loop=True):
        super(DropEdgeWrapper, self).__init__()

        # Copy Node's information
        for key, value in graph_wrapper.node_feat.items():
            self.node_feat_tensor_dict[key] = value

        self._num_nodes = graph_wrapper.num_nodes
        self._graph_lod = graph_wrapper.graph_lod
        self._num_graph = graph_wrapper.num_graph

        # Dropout Edges
        src, dst = graph_wrapper.edges
        u = L.uniform_random(
            shape=L.cast(L.shape(src), 'int64'), min=0., max=1.)

        # Avoid Empty Edges
        keeped = L.cast(u > dropout, dtype="float32")
        self._num_edges = L.reduce_sum(L.cast(keeped, "int32"))
        keeped = keeped + L.cast(self._num_edges == 0, dtype="float32")

        if keep_self_loop:
            self_loop = L.cast(src == dst, dtype="float32")
            keeped = keeped + self_loop

        keeped = (keeped > 0.5)
        src = paddle_helper.masked_select(src, keeped)
        dst = paddle_helper.masked_select(dst, keeped)
        src.stop_gradient = True
        dst.stop_gradient = True
        self._edges_src = src
        self._edges_dst = dst

        for key, value in graph_wrapper.edge_feat.items():
            self.edge_feat_tensor_dict[key] = paddle_helper.masked_select(
                value, keeped)

        self._edge_uniq_dst, _, uniq_count = L.unique_with_counts(
            dst, dtype="int32")
        self._edge_uniq_dst.stop_gradient = True
        last = L.reduce_sum(uniq_count, keep_dim=True)
        uniq_count = L.cumsum(uniq_count, exclusive=True)
        self._edge_uniq_dst_count = L.concat([uniq_count, last])
        self._edge_uniq_dst_count.stop_gradient = True
        self._indegree = get_degree(self._edges_dst, self._num_nodes)


class BatchGraphWrapper(BaseGraphWrapper):
    """Implement a graph wrapper that user can use their own data holder. 
    And this graph wrapper support multiple graphs which is benefit for data parallel algorithms.

    Args:
        num_nodes (int32 or int64): Shape [ num_graph ]. 

        num_edges (int32 or int64): Shape [ num_graph ]. 

        edges (int32 or int64): Shape [ total_num_edges_in_the_graphs, 2 ] 
                                  or Tuple with (src, dst).
   
        node_feats: A dictionary for node features. Each value should be tensor
                    with shape [ total_num_nodes_in_the_graphs, feature_size]

        edge_feats: A dictionary for edge features. Each value should be tensor
                    with shape [ total_num_edges_in_the_graphs, feature_size]

    """

    def __init__(self,
                 num_nodes,
                 num_edges,
                 edges,
                 node_feats=None,
                 edge_feats=None):
        super(BatchGraphWrapper, self).__init__()

        node_shift, edge_lod = self.__build_meta_data(num_nodes, num_edges)
        self.__build_edges(edges, node_shift, edge_lod, edge_feats)

        # assign node features
        if node_feats is not None:
            for key, value in node_feats.items():
                self.node_feat_tensor_dict[key] = value

        # other meta-data 
        self._edge_uniq_dst, _, uniq_count = L.unique_with_counts(
            self._edges_dst, dtype="int32")
        self._edge_uniq_dst.stop_gradient = True
        last = L.reduce_sum(uniq_count, keep_dim=True)
        uniq_count = L.cumsum(uniq_count, exclusive=True)
        self._edge_uniq_dst_count = L.concat([uniq_count, last])
        self._edge_uniq_dst_count.stop_gradient = True
        self._indegree = get_degree(self._edges_dst, self._num_nodes)

    def __build_meta_data(self, num_nodes, num_edges):
        """ Merge information for nodes and edges.
        """
        num_nodes = L.reshape(num_nodes, [-1])
        num_edges = L.reshape(num_edges, [-1])
        num_nodes = paddle_helper.ensure_dtype(num_nodes, dtype="int32")
        num_edges = paddle_helper.ensure_dtype(num_edges, dtype="int32")

        num_graph = L.shape(num_nodes)[0]
        sum_num_nodes = L.reduce_sum(num_nodes)
        sum_num_edges = L.reduce_sum(num_edges)
        edge_lod = L.concat(
            [L.cumsum(
                num_edges, exclusive=True), sum_num_edges])
        edge_lod = paddle_helper.lod_remove(edge_lod)

        node_shift = L.cumsum(num_nodes, exclusive=True)
        graph_lod = L.concat([node_shift, sum_num_nodes])
        graph_lod = paddle_helper.lod_remove(graph_lod)
        self._num_nodes = sum_num_nodes
        self._num_edges = sum_num_edges
        self._num_graph = num_graph
        self._graph_lod = graph_lod
        return node_shift, edge_lod

    def __build_edges(self, edges, node_shift, edge_lod, edge_feats):
        """ Merge subgraph edges. 
        """
        if isinstance(edges, tuple):
            src, dst = edges
        else:
            src = edges[:, 0]
            dst = edges[:, 1]

        src = L.reshape(src, [-1])
        dst = L.reshape(dst, [-1])
        src = paddle_helper.ensure_dtype(src, dtype="int32")
        dst = paddle_helper.ensure_dtype(dst, dtype="int32")
        # preprocess edges
        lod_dst = L.lod_reset(dst, edge_lod)
        node_shift = L.reshape(node_shift, [-1, 1])
        node_shift = L.sequence_expand_as(node_shift, lod_dst)
        node_shift = L.reshape(node_shift, [-1])
        src = src + node_shift
        dst = dst + node_shift
        # sort edges
        self._edges_dst, index = L.argsort(dst)
        self._edges_src = L.gather(src, index, overwrite=False)

        # assign edge features
        if edge_feats is not None:
            for key, efeat in edge_feats.items():
                self.edge_feat_tensor_dict[key] = L.gather(
                    efeat, index, overwrite=False)
