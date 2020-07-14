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

import numpy as np
from pgl.utils.mp_mapper import mp_reader_mapper
from sample import graph_saint_hetero, k_hop_sampler


def dataloader(source_node, label, batch_size=1024):
    index = np.arange(len(source_node))
    np.random.shuffle(index)

    def loader():
        start = 0
        while start < len(source_node):
            end = min(start + batch_size, len(source_node))
            yield source_node[index[start:end]], label[index[start:end]]
            start = end

    return loader


def sample_loader(args, phase, homograph, hetergraph, gw, source_node, label):
    if phase == 'train':
        sample_func = graph_saint_hetero
        batch_size = args.batch_size
        # max_depth for deepwalk
        other_args = len(args.test_samples)
    else:
        sample_func = k_hop_sampler
        batch_size = args.test_batch_size
        # sample num for k_hop
        other_args = args.test_samples

    def map_fun(node_label):
        node, label = node_label
        subgraph, train_index, sample_nodes, train_label = sample_func(
            homograph, hetergraph, node, other_args)
        feed_dict = gw.to_feed(subgraph)
        feed_dict['label'] = label if train_label is None else train_label
        feed_dict['train_index'] = train_index
        feed_dict['sub_node_index'] = sample_nodes
        return feed_dict

    loader = dataloader(source_node, label, batch_size)
    reader = mp_reader_mapper(
        loader, func=map_fun, num_works=args.sample_workers)

    for feed_dict in reader():
        yield feed_dict
