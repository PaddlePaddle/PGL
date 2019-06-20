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
from pgl.utils.logger import log
import train
import time


def node_batch_iter(nodes, node_label, batch_size):
    perm = np.arange(len(nodes))
    np.random.shuffle(perm)
    start = 0
    while start < len(nodes):
        index = perm[start:start + batch_size]
        start += batch_size
        yield nodes[index], node_label[index]


def traverse(item):
    if isinstance(item, list) or isinstance(item, np.ndarray):
        for i in iter(item):
            for j in traverse(i):
                yield j
    else:
        yield item


def flat_node_and_edge(nodes, eids):
    nodes = list(set(traverse(nodes)))
    eids = list(set(traverse(eids)))
    return nodes, eids


def worker(batch_info, graph, samples):
    def work():
        for batch_train_samples, batch_train_labels in batch_info:
            start_nodes = batch_train_samples
            nodes = start_nodes
            eids = []
            for max_deg in samples:
                pred, pred_eid = graph.sample_predecessor(
                    start_nodes, max_degree=max_deg, return_eids=True)
                last_nodes = nodes
                nodes = [nodes, pred]
                eids = [eids, pred_eid]
                nodes, eids = flat_node_and_edge(nodes, eids)
                # Find new nodes
                start_nodes = list(set(nodes) - set(last_nodes))
                if len(start_nodes) == 0:
                    break

            feed_dict = {}
            feed_dict["nodes"] = [int(n) for n in nodes]
            feed_dict["eids"] = [int(e) for e in eids]
            feed_dict["node_label"] = [int(n) for n in batch_train_labels]
            feed_dict["node_index"] = [int(n) for n in batch_train_samples]
            yield feed_dict

    return work


def multiprocess_graph_reader(graph,
                              graph_wrapper,
                              samples,
                              node_index,
                              batch_size,
                              node_label,
                              num_workers=4):
    def parse_to_subgraph(rd):
        def work():
            for data in rd():
                nodes = data["nodes"]
                eids = data["eids"]
                batch_train_labels = data["node_label"]
                batch_train_samples = data["node_index"]
                subgraph = graph.subgraph(nodes=nodes, eid=eids)
                sub_node_index = subgraph.reindex_from_parrent_nodes(
                    batch_train_samples)
                feed_dict = graph_wrapper.to_feed(subgraph)
                feed_dict["node_label"] = np.expand_dims(
                    np.array(
                        batch_train_labels, dtype="int64"), -1)
                feed_dict["node_index"] = sub_node_index
                yield feed_dict

        return work

    def reader():
        batch_info = list(
            node_batch_iter(
                node_index, node_label, batch_size=batch_size))
        block_size = int(len(batch_info) / num_workers + 1)
        reader_pool = []
        for i in range(num_workers):
            reader_pool.append(
                worker(batch_info[block_size * i:block_size * (i + 1)], graph,
                       samples))
        multi_process_sample = paddle.reader.multiprocess_reader(
            reader_pool, use_pipe=False)
        r = parse_to_subgraph(multi_process_sample)
        return paddle.reader.buffered(r, 1000)

    return reader()


def graph_reader(graph, graph_wrapper, samples, node_index, batch_size,
                 node_label):
    def reader():
        for batch_train_samples, batch_train_labels in node_batch_iter(
                node_index, node_label, batch_size=batch_size):
            start_nodes = batch_train_samples
            nodes = start_nodes
            eids = []
            for max_deg in samples:
                pred, pred_eid = graph.sample_predecessor(
                    start_nodes, max_degree=max_deg, return_eids=True)
                last_nodes = nodes
                nodes = [nodes, pred]
                eids = [eids, pred_eid]
                nodes, eids = flat_node_and_edge(nodes, eids)
                # Find new nodes
                start_nodes = list(set(nodes) - set(last_nodes))
                if len(start_nodes) == 0:
                    break

            subgraph = graph.subgraph(nodes=nodes, eid=eids)
            feed_dict = graph_wrapper.to_feed(subgraph)
            sub_node_index = subgraph.reindex_from_parrent_nodes(
                batch_train_samples)

            feed_dict["node_label"] = np.expand_dims(
                np.array(
                    batch_train_labels, dtype="int64"), -1)
            feed_dict["node_index"] = np.array(sub_node_index, dtype="int32")
            yield feed_dict

    return paddle.reader.buffered(reader, 1000)
