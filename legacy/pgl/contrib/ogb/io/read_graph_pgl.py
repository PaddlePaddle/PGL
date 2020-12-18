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
from ogb.io.read_graph_raw import read_csv_graph_raw


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


if __name__ == "__main__":
    # graph_list = read_csv_graph_dgl('dataset/proteinfunc_v2/raw', add_inverse_edge = True)
    graph_list = read_csv_graph_pgl(
        'dataset/ogbn_proteins_pgl/raw', add_inverse_edge=True)
    print(graph_list)
