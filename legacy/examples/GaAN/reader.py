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
import numpy as np
import pickle as pkl
import paddle
import paddle.fluid as fluid
import pgl
import time
from pgl.utils import mp_reader
from pgl.utils.logger import log
import time
import copy


def node_batch_iter(nodes, node_label, batch_size):
    """node_batch_iter
    """
    perm = np.arange(len(nodes))
    np.random.shuffle(perm)
    start = 0
    while start < len(nodes):
        index = perm[start:start + batch_size]
        start += batch_size
        yield nodes[index], node_label[index]


def traverse(item):
    """traverse
    """
    if isinstance(item, list) or isinstance(item, np.ndarray):
        for i in iter(item):
            for j in traverse(i):
                yield j
    else:
        yield item


def flat_node_and_edge(nodes):
    """flat_node_and_edge
    """
    nodes = list(set(traverse(nodes)))
    return nodes


def worker(batch_info, graph, graph_wrapper, samples):
    """Worker
    """

    def work():
        """work
        """
        _graph_wrapper = copy.copy(graph_wrapper)
        _graph_wrapper.node_feat_tensor_dict = {}
        for batch_train_samples, batch_train_labels in batch_info:
            start_nodes = batch_train_samples
            nodes = start_nodes
            edges = []
            for max_deg in samples:
                pred_nodes = graph.sample_predecessor(
                    start_nodes, max_degree=max_deg)

                for dst_node, src_nodes in zip(start_nodes, pred_nodes):
                    for src_node in src_nodes:
                        edges.append((src_node, dst_node))

                last_nodes = nodes
                nodes = [nodes, pred_nodes]
                nodes = flat_node_and_edge(nodes)
                # Find new nodes
                start_nodes = list(set(nodes) - set(last_nodes))
                if len(start_nodes) == 0:
                    break

            subgraph = graph.subgraph(
                nodes=nodes,
                edges=edges,
                with_node_feat=True,
                with_edge_feat=True)

            sub_node_index = subgraph.reindex_from_parrent_nodes(
                batch_train_samples)
            
            feed_dict = _graph_wrapper.to_feed(subgraph)
            
            feed_dict["node_label"] = batch_train_labels
            feed_dict["node_index"] = sub_node_index
            feed_dict["parent_node_index"] = np.array(nodes, dtype="int64")
            yield feed_dict

    return work


def multiprocess_graph_reader(graph,
                              graph_wrapper,
                              samples,
                              node_index,
                              batch_size,
                              node_label,
                              with_parent_node_index=False,
                              num_workers=4):
    """multiprocess_graph_reader
    """

    def parse_to_subgraph(rd, prefix, node_feat, _with_parent_node_index):
        """parse_to_subgraph
        """

        def work():
            """work
            """
            for data in rd():
                feed_dict = data
                for key in node_feat:
                    feed_dict[prefix + '/node_feat/' + key] = node_feat[key][
                        feed_dict["parent_node_index"]]
                if not _with_parent_node_index:
                    del feed_dict["parent_node_index"]
                yield feed_dict

        return work

    def reader():
        """reader"""
        batch_info = list(
            node_batch_iter(
                node_index, node_label, batch_size=batch_size))
        block_size = int(len(batch_info) / num_workers + 1)
        reader_pool = []
        for i in range(num_workers):
            reader_pool.append(
                worker(batch_info[block_size * i:block_size * (i + 1)], graph,
                       graph_wrapper, samples))

        if len(reader_pool) == 1:
            r = parse_to_subgraph(reader_pool[0],
                                  repr(graph_wrapper), graph.node_feat,
                                  with_parent_node_index)
        else:
            multi_process_sample = mp_reader.multiprocess_reader(
                reader_pool, use_pipe=True, queue_size=1000)
            r = parse_to_subgraph(multi_process_sample,
                                  repr(graph_wrapper), graph.node_feat,
                                  with_parent_node_index)
        return paddle.reader.buffered(r, num_workers)
    
    return reader()
