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
"""pgl read_csv_graph for ogb
"""

import pandas as pd
import os.path as osp
import numpy as np
import pgl
from pgl import heter_graph
from ogb.io.read_graph_raw import read_csv_graph_raw, read_csv_heterograph_raw
from collections import OrderedDict
import logging
logger = logging.getLogger(__name__)


def read_csv_graph_pgl(raw_dir, add_inverse_edge=False):
    """Read CSV data and build PGL Graph
    """
    graph_list = read_csv_graph_raw(raw_dir, add_inverse_edge)
    pgl_graph_list = []

    for graph in graph_list:
        edges = list(zip(graph["edge_index"][0], graph["edge_index"][1]))
        g = pgl.graph.Graph(num_nodes=graph["num_nodes"], edges=edges)

        if graph["edge_feat"] is not None:
            g.edge_feat["feat"] = graph["edge_feat"]

        if graph["node_feat"] is not None:
            g.node_feat["feat"] = graph["node_feat"]

        pgl_graph_list.append(g)

    return pgl_graph_list


def read_csv_heterograph_pgl(raw_dir,
                             add_inverse_edge=False,
                             additional_node_files=[],
                             additional_edge_files=[]):
    """Read CSV data and build PGL heterograph
    """
    graph_list = read_csv_heterograph_raw(
        raw_dir,
        add_inverse_edge,
        additional_node_files=additional_node_files,
        additional_edge_files=additional_edge_files)
    pgl_graph_list = []

    logger.info('Converting graphs into PGL objects...')

    for graph in graph_list:
        # logger.info(graph)
        node_index = OrderedDict()
        node_types = []
        num_nodes = 0
        for k, v in graph["num_nodes_dict"].items():
            node_types.append(
                np.ones(
                    shape=[v, 1], dtype='int64') * len(node_index))
            node_index[k] = (v, num_nodes)
            num_nodes += v
        # logger.info(node_index)
        node_types = np.vstack(node_types)
        edges_by_types = {}
        for k, v in graph["edge_index_dict"].items():
            v[0, :] += node_index[k[0]][1]
            v[1, :] += node_index[k[2]][1]
            inverse_v = np.array(v)
            inverse_v[0, :] = v[1, :]
            inverse_v[1, :] = v[0, :]
            if k[0] != k[2]:
                edges_by_types["{}2{}".format(k[0][0], k[2][0])] = v.T
                edges_by_types["{}2{}".format(k[2][0], k[0][0])] = inverse_v.T
            else:
                edges = np.hstack((v, inverse_v))
                edges_by_types["{}2{}".format(k[0][0], k[2][0])] = edges.T

        node_features = {
            'index': np.array([i for i in range(num_nodes)]).reshape(
                -1, 1).astype(np.int64)
        }
        # logger.info(edges_by_types.items())
        g = heter_graph.HeterGraph(
            num_nodes=num_nodes,
            edges=edges_by_types,
            node_types=node_types,
            node_feat=node_features)
        g.edge_feat_dict = graph['edge_feat_dict']
        g.node_feat_dict = graph['node_feat_dict']
        g.num_node_dict = node_index
        pgl_graph_list.append(g)

    logger.info("Done, converted!")
    return pgl_graph_list


if __name__ == "__main__":
    # graph_list = read_csv_graph_dgl('dataset/proteinfunc_v2/raw', add_inverse_edge = True)
    graph_list = read_csv_graph_pgl(
        'dataset/ogbn_proteins_pgl/raw', add_inverse_edge=True)
    print(graph_list)
