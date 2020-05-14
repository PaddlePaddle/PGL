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
import os
import sys
import numpy as np
import pickle as pkl
import paddle
import paddle.fluid as fluid
import socket
import pgl
import time

from pgl.utils import mp_reader
from pgl.utils.logger import log
from pgl import redis_graph


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


def flat_node_and_edge(nodes, eids):
    """flat_node_and_edge
    """
    nodes = list(set(traverse(nodes)))
    eids = list(set(traverse(eids)))
    return nodes, eids


def worker(batch_info, graph_wrapper, samples):
    """Worker
    """

    def work():
        """work
        """
        redis_configs = [{
            "host": socket.gethostbyname(socket.gethostname()),
            "port": 7430
        }, ]
        graph = redis_graph.RedisGraph("sub_graph", redis_configs, 64)
        first = True
        for batch_train_samples, batch_train_labels in batch_info:
            start_nodes = batch_train_samples
            nodes = start_nodes
            eids = []
            eid2edges = {}
            for max_deg in samples:
                pred, pred_eid = graph.sample_predecessor(
                    start_nodes, max_degree=max_deg, return_eids=True)
                for _dst, _srcs, _eids in zip(start_nodes, pred, pred_eid):
                    for _src, _eid in zip(_srcs, _eids):
                        eid2edges[_eid] = (_src, _dst)

                last_nodes = nodes
                nodes = [nodes, pred]
                eids = [eids, pred_eid]
                nodes, eids = flat_node_and_edge(nodes, eids)
                # Find new nodes
                start_nodes = list(set(nodes) - set(last_nodes))
                if len(start_nodes) == 0:
                    break

            subgraph = graph.subgraph(nodes=nodes, eid=eids, edges=[ eid2edges[e] for e in eids])
            sub_node_index = subgraph.reindex_from_parrent_nodes(
                batch_train_samples)
            feed_dict = graph_wrapper.to_feed(subgraph)
            feed_dict["node_label"] = np.expand_dims(
                np.array(
                    batch_train_labels, dtype="int64"), -1)
            feed_dict["node_index"] = sub_node_index
            yield feed_dict

    return work


def multiprocess_graph_reader(
                              graph_wrapper,
                              samples,
                              node_index,
                              batch_size,
                              node_label,
                              num_workers=4):
    """multiprocess_graph_reader
    """

    def parse_to_subgraph(rd):
        """parse_to_subgraph
        """

        def work():
            """work
            """
            last = time.time()
            for data in rd():
                this = time.time()
                feed_dict = data
                now = time.time()
                last = now
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
                worker(batch_info[block_size * i:block_size * (i + 1)], 
                       graph_wrapper, samples))
        multi_process_sample = mp_reader.multiprocess_reader(
            reader_pool, use_pipe=True, queue_size=1000)
        r = parse_to_subgraph(multi_process_sample)
        return paddle.reader.buffered(r, 1000)

    return reader()


def load_data():
    """
        data from https://github.com/matenure/FastGCN/issues/8
        reddit.npz: https://drive.google.com/open?id=19SphVl_Oe8SJ1r87Hr5a6znx3nJu1F2J
        reddit_index_label is preprocess from reddit.npz without feats key.
    """
    data_dir = os.path.dirname(os.path.abspath(__file__))
    data = np.load(os.path.join(data_dir, "data/reddit_index_label.npz"))

    num_class = 41

    train_label = data['y_train']
    val_label = data['y_val']
    test_label = data['y_test']

    train_index = data['train_index']
    val_index = data['val_index']
    test_index = data['test_index']

    return {
        "train_index": train_index,
        "train_label": train_label,
        "val_label": val_label,
        "val_index": val_index,
        "test_index": test_index,
        "test_label": test_label,
        "num_class": 41
    }

def get_iter(args, graph_wrapper, mode):
    data = load_data()
    train_iter = multiprocess_graph_reader(
        graph_wrapper,
        samples=args.samples,
        num_workers=args.num_sample_workers,
        batch_size=args.batch_size,
        node_index=data['train_index'],
        node_label=data["train_label"])
    return train_iter

if __name__ == '__main__':
    for e in train_iter():
        print(e)

