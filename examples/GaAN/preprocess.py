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
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from ogb.nodeproppred import NodePropPredDataset, Evaluator

import pgl
import numpy as np
import os
import time


def get_graph_data(d_name="ogbn-proteins", mini_data=False):
    """
        Param:
            d_name: name of dataset
            mini_data: if mini_data==True, only use a small dataset (for test)
    """
    # import ogb data
    dataset = NodePropPredDataset(name = d_name)
    num_tasks = dataset.num_tasks # obtaining the number of prediction tasks in a dataset

    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    graph, label = dataset[0]
    
    # reshape
    graph["edge_index"] = graph["edge_index"].T
    
    # mini dataset
    if mini_data: 
        graph['num_nodes'] = 500
        mask = (graph['edge_index'][:, 0] < 500)*(graph['edge_index'][:, 1] < 500)
        graph["edge_index"] = graph["edge_index"][mask]
        graph["edge_feat"] = graph["edge_feat"][mask]
        label = label[:500]
        train_idx = np.arange(0,400)
        valid_idx = np.arange(400,450)
        test_idx = np.arange(450,500)
    

    
    # read/compute node feature
    if mini_data:
        node_feat_path = './dataset/ogbn_proteins_node_feat_small.npy'
    else:
        node_feat_path = './dataset/ogbn_proteins_node_feat.npy'

    new_node_feat = None
    if os.path.exists(node_feat_path):
        print("Begin: read node feature".center(50, '='))
        new_node_feat = np.load(node_feat_path)
        print("End: read node feature".center(50, '='))
    else:
        print("Begin: compute node feature".center(50, '='))
        start = time.perf_counter()
        for i in range(graph['num_nodes']):
            if i % 100 == 0:
                dur = time.perf_counter() - start
                print("{}/{}({}%), times: {:.2f}s".format(
                    i, graph['num_nodes'], i/graph['num_nodes']*100, dur
                ))
            mask = (graph['edge_index'][:, 0] == i)
            
            current_node_feat = np.mean(np.compress(mask, graph['edge_feat'], axis=0),
                                        axis=0, keepdims=True)
            if i == 0:
                new_node_feat = [current_node_feat]
            else:  
                new_node_feat.append(current_node_feat)

        new_node_feat = np.concatenate(new_node_feat, axis=0)
        print("End: compute node feature".center(50,'='))

        print("Saving node feature in "+node_feat_path.center(50, '='))
        np.save(node_feat_path, new_node_feat)
        print("Saving finish".center(50,'='))
    
    print(new_node_feat)
    
    
    # create graph
    g = pgl.graph.Graph(
        num_nodes=graph["num_nodes"],
        edges = graph["edge_index"],
        node_feat = {'node_feat': new_node_feat},
        edge_feat = None
    )
    print("Create graph")
    print(g)
    return g, label, train_idx, valid_idx, test_idx, Evaluator(d_name)
    