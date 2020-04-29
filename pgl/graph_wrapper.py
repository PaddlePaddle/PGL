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

__all__ = ["BaseGraphWrapper", "GraphWrapper", "StaticGraphWrapper"]


def send(src, dst, nfeat, efeat, message_func):
    """Send message from src to dst.
    """
    src_feat = op.read_rows(nfeat, src)
    dst_feat = op.read_rows(nfeat, dst)
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
            init_output = fluid.layers.fill_constant(
                shape=[num_nodes, out_dim], value=0, dtype=msg.dtype)
            init_output.stop_gradient = False
            empty_msg_flag = fluid.layers.cast(num_edges > 0, dtype=msg.dtype)
            msg = msg * empty_msg_flag
            output = paddle_helper.scatter_add(init_output, dst, msg)
            return output
        except TypeError as e:
            warnings.warn(
                "scatter_add is not supported with paddle version <= 1.5")

            def sum_func(message):
                return fluid.layers.sequence_pool(message, "sum")

            reduce_function = sum_func

    bucketed_msg = op.nested_lod_reset(msg, bucketing_index)
    output = reduce_function(bucketed_msg)
    output_dim = output.shape[-1]

    empty_msg_flag = fluid.layers.cast(num_edges > 0, dtype=output.dtype)
    output = output * empty_msg_flag

    init_output = fluid.layers.fill_constant(
        shape=[num_nodes, output_dim], value=0, dtype=output.dtype)
    init_output.stop_gradient = True
    final_output = fluid.layers.scatter(init_output, uniq_dst, output)
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
        self._node_ids = None
        self._graph_lod = None
        self._num_graph = None
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

        node_ids_value = np.arange(0, graph.num_nodes, dtype="int64")
        self._node_ids, init = paddle_helper.constant(
            name=self._data_name_prefix + "/node_ids",
            dtype="int64",
            value=node_ids_value)
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
    that attributes and features in the graph are :code:`fluid.layers.data`.
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
        self._num_edges = fluid.layers.data(
            self._data_name_prefix + '/num_edges',
            shape=[1],
            append_batch_size=False,
            dtype="int64",
            stop_gradient=True)
        self._num_graph = fluid.layers.data(
            self._data_name_prefix + '/num_graph',
            shape=[1],
            append_batch_size=False,
            dtype="int64",
            stop_gradient=True)
        self._edges_src = fluid.layers.data(
            self._data_name_prefix + '/edges_src',
            shape=[None],
            append_batch_size=False,
            dtype="int64",
            stop_gradient=True)
        self._edges_dst = fluid.layers.data(
            self._data_name_prefix + '/edges_dst',
            shape=[None],
            append_batch_size=False,
            dtype="int64",
            stop_gradient=True)
        self._num_nodes = fluid.layers.data(
            self._data_name_prefix + '/num_nodes',
            shape=[1],
            append_batch_size=False,
            dtype='int64',
            stop_gradient=True)

        self._edge_uniq_dst = fluid.layers.data(
            self._data_name_prefix + "/uniq_dst",
            shape=[None],
            append_batch_size=False,
            dtype="int64",
            stop_gradient=True)

        self._graph_lod = fluid.layers.data(
            self._data_name_prefix + "/graph_lod",
            shape=[None],
            append_batch_size=False,
            dtype="int32",
            stop_gradient=True)

        self._edge_uniq_dst_count = fluid.layers.data(
            self._data_name_prefix + "/uniq_dst_count",
            shape=[None],
            append_batch_size=False,
            dtype="int32",
            stop_gradient=True)

        self._node_ids = fluid.layers.data(
            self._data_name_prefix + "/node_ids",
            shape=[None],
            append_batch_size=False,
            dtype="int64",
            stop_gradient=True)
        self._indegree = fluid.layers.data(
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
            self._node_ids,
            self._indegree,
            self._graph_lod,
            self._num_graph,
            self._num_edges,
        ])

    def __create_graph_node_feat_holders(self, node_feat_name, node_feat_shape,
                                         node_feat_dtype):
        """Create data holders for node features.
        """
        feat_holder = fluid.layers.data(
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
        feat_holder = fluid.layers.data(
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
        feed_dict[self._data_name_prefix + '/node_ids'] = graph.nodes
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
