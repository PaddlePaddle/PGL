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
"""
    Dataset and DataLoader for GNNAutoScale.
"""

import os
import numpy as np
from functools import partial

import paddle
from pgl.utils.logger import log
from pgl.utils.data import Dataset
from pgl.utils.data.dataloader import Dataloader
from pgl.sampling.custom import subgraph


class SubgraphData(object):
    def __init__(self, subgraph, batch_size, nodes, offset, count):
        self.subgraph = subgraph
        self.batch_size = batch_size
        self.nodes = nodes
        self.offset = offset
        self.count = count


class PartitionDataset(Dataset):
    """PartitionDataset helps build train dataset, which can be used to 
       build eval and test dataset too.
    """

    def __init__(self, split):
        self.split = split

    def __getitem__(self, idx):
        return (idx, np.arange(self.split[idx], self.split[idx + 1]))

    def __len__(self):
        return len(self.split) - 1


class EvalPartitionDataset(Dataset):
    """EvalPartitionDataset helps build eval and test dataset, which will be generated 
       in advance for better validation/test speed. 
    """

    def __init__(self, graph, split, batch_size, flag_buffer):
        self.split = split[::batch_size]

        num_nodes = graph.num_nodes
        if self.split[-1] != num_nodes:
            self.split = paddle.concat([
                self.split, paddle.to_tensor(
                    [num_nodes], dtype=self.split.dtype)
            ])
        batches_nodes = np.split(np.arange(num_nodes), self.split[1:-1])
        batches_nodes = [(i, batches_nodes[i])
                         for i in range(len(batches_nodes))]
        collate_fn = partial(
            subdata_batch_fn,
            graph=graph,
            split=self.split,
            flag_buffer=flag_buffer)
        self.data_list = list(
            Dataloader(
                dataset=batches_nodes,
                batch_size=1,
                num_workers=5,
                shuffle=False,
                collate_fn=collate_fn))

    def __getitem__(self, idx):
        return self.data_list[idx]

    def __len__(self):
        return len(self.split) - 1


def one_hop_neighbor(graph, nodes, flag_buffer):
    """Find one hop neighbors.
    """
    pred_nodes, pred_eids = graph.predecessor(nodes, return_eids=True)
    pred_nodes = np.concatenate(pred_nodes, -1)
    pred_eids = np.concatenate(pred_eids, -1)

    flag_buffer[nodes] = 1
    out_of_batch_neighbors = pred_nodes[flag_buffer[pred_nodes] == 0]
    out_of_batch_neighbors = np.unique(out_of_batch_neighbors)
    new_nodes = np.concatenate((nodes, out_of_batch_neighbors))
    flag_buffer[nodes] = 0
    return new_nodes, pred_eids


def subdata_batch_fn(batches_nodes, graph, split, flag_buffer):
    """Basic function for creating batch subgraph data.
    """
    batch_ids, nodes = zip(*batches_nodes)
    batch_ids = np.array(batch_ids)
    orig_nodes = np.concatenate(nodes, axis=0)

    new_nodes, pred_eids = one_hop_neighbor(graph, orig_nodes, flag_buffer)
    sub_graph = subgraph(graph, nodes=new_nodes, eid=pred_eids)
    batch_size = np.size(orig_nodes)
    offset = split[batch_ids]
    count = split[batch_ids + 1] - split[batch_ids]

    return SubgraphData(sub_graph, batch_size, new_nodes, offset, count)
