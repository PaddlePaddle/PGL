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
"""reader.py"""
import os
import numpy as np
import pickle as pkl
import paddle
import paddle.fluid as fluid
import pgl
import time
from pgl.utils.logger import log
from pgl.utils import mp_reader


def batch_iter(data, batch_size):
    """batch_iter"""
    src, dst, eid = data
    perm = np.arange(len(eid))
    np.random.shuffle(perm)
    start = 0
    while start < len(src):
        index = perm[start:start + batch_size]
        start += batch_size
        yield src[index], dst[index], eid[index]


def traverse(item):
    """traverse"""
    if isinstance(item, list) or isinstance(item, np.ndarray):
        for i in iter(item):
            for j in traverse(i):
                yield j
    else:
        yield item


def flat_node_and_edge(nodes, eids):
    """flat_node_and_edge"""
    nodes = list(set(traverse(nodes)))
    eids = list(set(traverse(eids)))
    return nodes, eids


def graph_reader(num_layers,
                 graph_wrappers,
                 data,
                 batch_size,
                 samples,
                 num_workers,
                 feed_name_list,
                 use_pyreader=False,
                 graph=None,
                 predict=False):
    """graph_reader
    """
    assert num_layers == len(samples), "Must be unified number of layers!"
    if num_workers > 1:
        return multiprocess_graph_reader(
            num_layers,
            graph_wrappers,
            data,
            batch_size,
            samples,
            num_workers,
            feed_name_list,
            use_pyreader,
            graph=graph,
            predict=predict)

    batch_info = list(batch_iter(data, batch_size=batch_size))
    work = worker(
        num_layers,
        batch_info,
        graph_wrappers,
        samples,
        feed_name_list,
        use_pyreader,
        graph=graph,
        predict=predict)

    def reader():
        """reader"""
        for batch in work():
            yield batch

    return reader
    #return paddle.reader.buffered(reader, 100)


def worker(num_layers, batch_info, graph_wrappers, samples, feed_name_list,
           use_pyreader, graph, predict):
    """worker
    """
    pid = os.getppid()
    np.random.seed((int(time.time() * 10000) + pid) % 65535)

    graphs = [graph, graph]

    def work():
        """work
        """
        feed_dict = {}
        ind = 0
        perm = np.arange(0, len(batch_info))
        np.random.shuffle(perm)
        for p in perm:
            batch_src, batch_dst, batch_eid = batch_info[p]
            ind += 1
            ind_start = time.time()
            try:
                nodes = start_nodes = np.concatenate([batch_src, batch_dst], 0)
                eids = []
                layer_nodes, layer_eids = [], []
                for layer_idx in reversed(range(num_layers)):
                    if len(start_nodes) == 0:
                        layer_nodes = [nodes] + layer_nodes
                        layer_eids = [eids] + layer_eids
                        continue
                    pred_nodes, pred_eids = graphs[
                        layer_idx].sample_predecessor(
                            start_nodes, samples[layer_idx], return_eids=True)
                    last_nodes = nodes
                    nodes, eids = flat_node_and_edge([nodes, pred_nodes],
                                                     [eids, pred_eids])
                    layer_nodes = [nodes] + layer_nodes
                    layer_eids = [eids] + layer_eids
                    # Find new nodes
                    start_nodes = list(set(nodes) - set(last_nodes))
                if predict is False:
                    eids = (batch_eid * 2 + 1).tolist() + (batch_eid * 2
                                                           ).tolist()
                    layer_eids[0] = list(set(layer_eids[0]) - set(eids))

                # layer_nodes[0]: use first layer nodes as all subgraphs' nodes
                subgraph = graphs[0].subgraph(
                    nodes=layer_nodes[0], eid=layer_eids[0])
                node_feat = np.array(layer_nodes[0], dtype="int64")
                subgraph.node_feat["index"] = node_feat

            except Exception as e:
                print(e)
                if len(feed_dict) > 0:
                    yield feed_dict
                continue
            feed_dict = graph_wrappers[0].to_feed(subgraph)

            # only reindex from first subgraph
            sub_src_idx = subgraph.reindex_from_parrent_nodes(batch_src)
            sub_dst_idx = subgraph.reindex_from_parrent_nodes(batch_dst)

            feed_dict["src_index"] = sub_src_idx.astype("int64")
            feed_dict["dst_index"] = sub_dst_idx.astype("int64")
            if predict:
                feed_dict["node_id"] = batch_src.astype("int64")

            if use_pyreader:
                yield [feed_dict[name] for name in feed_name_list]
            else:
                yield feed_dict

    return work


def multiprocess_graph_reader(num_layers, graph_wrappers, data, batch_size,
                              samples, num_workers, feed_name_list,
                              use_pyreader, graph, predict):
    """ multiprocess_graph_reader
    """

    def parse_to_subgraph(rd):
        """ parse_to_subgraph
        """

        def work():
            """ work
            """
            for data in rd():
                yield data

        return work

    def reader():
        """ reader
        """
        batch_info = list(batch_iter(data, batch_size=batch_size))
        log.info("The size of batch:%d" % (len(batch_info)))
        block_size = int(len(batch_info) / num_workers + 1)
        reader_pool = []
        for i in range(num_workers):
            reader_pool.append(
                worker(num_layers, batch_info[block_size * i:block_size * (
                    i + 1)], graph_wrappers, samples, feed_name_list,
                       use_pyreader, graph, predict))
        use_pipe = True
        multi_process_sample = mp_reader.multiprocess_reader(
            reader_pool, use_pipe=use_pipe)
        r = parse_to_subgraph(multi_process_sample)
        if use_pipe:
            return paddle.reader.buffered(r, 5 * num_workers)
        else:
            return r

    return reader()
