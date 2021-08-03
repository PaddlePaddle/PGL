# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import pgl


class EgoInfo(object):
    def __init__(self,
                 node_id=None,
                 feature=None,
                 edges=None,
                 edges_type=None,
                 edges_weight=None):
        self.node_id = node_id
        self.feature = feature
        self.edges = edges
        self.edges_type = edges_type
        self.edges_weight = edges_weight


def graphsage_sampling(graph, node_ids, samples, edge_types):
    """  Heter-EgoGraph                
                              ____ n1
                             /----n2
                     etype1 /---- n3
                   --------/-----n4
                  /        \-----n5
                 /
                /   
        start_n               
                \             /----n6
                 \  etype2 /---- n7
                   --------/-----n8
                          \-----n9

        TODO @Yelrose: Speed up and standarize to pgl.distributed.sampling
    """

    # All Nodes
    all_new_nodes = [node_ids]

    # Node Index
    all_new_nodes_index = [np.zeros_like(node_ids, dtype="int64")]

    # The Ego Index for each graph
    all_new_nodes_ego_index = [np.arange(0, len(node_ids), dtype="int64")]

    unique_nodes = set(node_ids)

    ego_graph_dict = [
        EgoInfo(
            node_id=[n], edges=[], edges_type=[], edges_weight=[])
        for n in node_ids
    ]

    for sample in samples:
        cur_node_ids = all_new_nodes[-1]
        cur_node_ego_index = all_new_nodes_ego_index[-1]
        cur_node_index = all_new_nodes_index[-1]

        nxt_node_ids = []
        nxt_node_ego_index = []
        nxt_node_index = []
        for edge_type_id, edge_type in enumerate(edge_types):
            cur_succs = graph.sample_successor(
                cur_node_ids, max_degree=sample, edge_type=edge_type)
            for succs, ego_index, parent_index in zip(
                    cur_succs, cur_node_ego_index, cur_node_index):
                if len(succs) == 0:
                    succs = [0]
                ego = ego_graph_dict[ego_index]

                for succ in succs:
                    unique_nodes.add(succ)
                    succ_index = len(ego.node_id)
                    ego.node_id.append(succ)
                    nxt_node_ids.append(succ)
                    nxt_node_ego_index.append(ego_index)
                    nxt_node_index.append(succ_index)
                    if succ == 0:
                        ego.edges.append((succ_index, succ_index))
                        ego.edges_type.append(edge_type_id)
                        ego.edges_weight.append(1.0 / len(succs))
                    else:
                        ego.edges.append((succ_index, parent_index))
                        ego.edges_type.append(edge_type_id)
                        ego.edges_weight.append(1.0 / len(succs))
        all_new_nodes.append(nxt_node_ids)
        all_new_nodes_index.append(nxt_node_index)
        all_new_nodes_ego_index.append(nxt_node_ego_index)

    pgl_ego_graph = []
    for ego in ego_graph_dict:
        pg = pgl.Graph(
            num_nodes=len(ego.node_id),
            edges=ego.edges,
            edge_feat={
                "edge_type": np.array(
                    ego.edges_type, dtype="int64"),
                "edge_weight": np.array(
                    ego.edges_weight, dtype="float32"),
            },
            node_feat={"node_id": np.array(
                ego.node_id, dtype="int64"), })
        pgl_ego_graph.append(pg)
    return pgl_ego_graph, list(unique_nodes)
