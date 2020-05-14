"""Graph Dataset
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import os
import pgl
import sys

import numpy as np

from pgl.utils.logger import log
from dataset.base_dataset import BaseDataGenerator
from pgl.sample import alias_sample
from pgl.sample import pinsage_sample
from pgl.sample import graphsage_sample 
from pgl.sample import edge_hash


class GraphGenerator(BaseDataGenerator):
    def __init__(self, graph_wrappers, data, batch_size, samples,
        num_workers, feed_name_list, use_pyreader,
        phase, graph_data_path, shuffle=True, buf_size=1000):

        super(GraphGenerator, self).__init__(
            buf_size=buf_size,
            num_workers=num_workers,
            batch_size=batch_size, shuffle=shuffle)
        # For iteration
        self.line_examples = data

        self.graph_wrappers = graph_wrappers
        self.samples = samples
        self.feed_name_list = feed_name_list
        self.use_pyreader = use_pyreader
        self.phase = phase
        self.load_graph(graph_data_path)
        self.num_layers = len(graph_wrappers)

    def load_graph(self, graph_data_path):
        self.graph = pgl.graph.MemmapGraph(graph_data_path)
        self.alias = np.load(os.path.join(graph_data_path, "alias.npy"), mmap_mode="r")
        self.events = np.load(os.path.join(graph_data_path, "events.npy"), mmap_mode="r")
        self.term_ids = np.load(os.path.join(graph_data_path, "term_ids.npy"), mmap_mode="r")
 
    def batch_fn(self, batch_ex):
        # batch_ex = [
        #     (src, dst, neg),
        #     (src, dst, neg),
        #     (src, dst, neg),
        #     ]
        #
        batch_src = []
        batch_dst = []
        batch_neg = []
        for batch in batch_ex:
            batch_src.append(batch[0])
            batch_dst.append(batch[1])
            if len(batch) == 3: # default neg samples
                batch_neg.append(batch[2])

        if len(batch_src) != self.batch_size:
            if self.phase == "train":
                return None  #Skip

        if len(batch_neg) > 0:
            batch_neg = np.unique(np.concatenate(batch_neg))
        batch_src = np.array(batch_src, dtype="int64")
        batch_dst = np.array(batch_dst, dtype="int64")

        sampled_batch_neg = alias_sample(batch_dst.shape, self.alias, self.events)
    
        if len(batch_neg) > 0:
            batch_neg = np.concatenate([batch_neg, sampled_batch_neg], 0)
        else:
            batch_neg = sampled_batch_neg

        if self.phase == "train":
            ignore_edges = np.concatenate([np.stack([batch_src, batch_dst], 1), np.stack([batch_dst, batch_src], 1)], 0)
        else:
            ignore_edges = set()

        nodes = np.unique(np.concatenate([batch_src, batch_dst, batch_neg], 0))
        subgraphs = graphsage_sample(self.graph, nodes, self.samples, ignore_edges=ignore_edges)
        #subgraphs[0].reindex_to_parrent_nodes(subgraphs[0].nodes)
        feed_dict = {}
        for i in range(self.num_layers):
            feed_dict.update(self.graph_wrappers[i].to_feed(subgraphs[i]))

        # only reindex from first subgraph
        sub_src_idx = subgraphs[0].reindex_from_parrent_nodes(batch_src)
        sub_dst_idx = subgraphs[0].reindex_from_parrent_nodes(batch_dst)
        sub_neg_idx = subgraphs[0].reindex_from_parrent_nodes(batch_neg)

        feed_dict["user_index"] = np.array(sub_src_idx, dtype="int64")
        feed_dict["item_index"] = np.array(sub_dst_idx, dtype="int64")
        feed_dict["neg_item_index"] = np.array(sub_neg_idx, dtype="int64")
        feed_dict["term_ids"] = self.term_ids[subgraphs[0].node_feat["index"]].astype(np.int64)
        return feed_dict

    def __call__(self):
        return self.generator()

    def generator(self):
        try:
            for feed_dict in super(GraphGenerator, self).generator():
                if self.use_pyreader:
                    yield [feed_dict[name] for name in self.feed_name_list]
                else:
                    yield feed_dict

        except Exception as e:
            log.exception(e)
 

    
