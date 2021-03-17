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
        phase, graph_data_path, shuffle=True, buf_size=1000, neg_type="batch_neg"):

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
        self.neg_type= neg_type

    def load_graph(self, graph_data_path):
        self.graph = pgl.graph.MemmapGraph(graph_data_path)
        self.alias = np.load(os.path.join(graph_data_path, "alias.npy"), mmap_mode="r")
        self.events = np.load(os.path.join(graph_data_path, "events.npy"), mmap_mode="r")
        #self.term_ids = np.load(os.path.join(graph_data_path, "term_ids.npy"), mmap_mode="r")
        self.term_ids = self.graph.node_feat["term_ids"]
 
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

        if self.neg_type == "batch_neg":
            batch_neg = batch_dst
        else:
            # TODO user define shape of neg_sample
            neg_shape = batch_dst.shape
            sampled_batch_neg = alias_sample(neg_shape, self.alias, self.events)
            batch_neg = np.concatenate([batch_neg, sampled_batch_neg], 0)

        if self.phase == "train":
            # TODO user define ignore edges or not
            #ignore_edges = np.concatenate([np.stack([batch_src, batch_dst], 1), np.stack([batch_dst, batch_src], 1)], 0)
            ignore_edges = set()
        else:
            ignore_edges = set()

        nodes = np.unique(np.concatenate([batch_src, batch_dst, batch_neg], 0))
        subgraphs = graphsage_sample(self.graph, nodes, self.samples, ignore_edges=ignore_edges)
        subgraphs[0].node_feat["index"] = subgraphs[0].reindex_to_parrent_nodes(subgraphs[0].nodes).astype(np.int64)
        subgraphs[0].node_feat["term_ids"] = self.term_ids[subgraphs[0].node_feat["index"]].astype(np.int64)
        feed_dict = {}
        for i in range(self.num_layers):
            feed_dict.update(self.graph_wrappers[i].to_feed(subgraphs[i]))

        # only reindex from first subgraph
        sub_src_idx = subgraphs[0].reindex_from_parrent_nodes(batch_src)
        sub_dst_idx = subgraphs[0].reindex_from_parrent_nodes(batch_dst)
        sub_neg_idx = subgraphs[0].reindex_from_parrent_nodes(batch_neg)

        feed_dict["user_index"] = np.array(sub_src_idx, dtype="int64")
        feed_dict["pos_item_index"] = np.array(sub_dst_idx, dtype="int64")
        feed_dict["neg_item_index"] = np.array(sub_neg_idx, dtype="int64")

        feed_dict["user_real_index"] = np.array(batch_src, dtype="int64")
        feed_dict["pos_item_real_index"] = np.array(batch_dst, dtype="int64")
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
 

    
class NodeClassificationGenerator(GraphGenerator):
    def batch_fn(self, batch_ex):
        # batch_ex = [
        #     (node, label),
        #     (node, label),
        #     ]
        #
        batch_node = []
        batch_label = []
        for batch in batch_ex:
            batch_node.append(batch[0])
            batch_label.append(batch[1])

        if len(batch_node) != self.batch_size:
            if self.phase == "train":
                return None  #Skip

        batch_node = np.array(batch_node, dtype="int64")
        batch_label = np.array(batch_label, dtype="int64")

        subgraphs = graphsage_sample(self.graph, batch_node, self.samples)
        subgraphs[0].node_feat["index"] = subgraphs[0].reindex_to_parrent_nodes(subgraphs[0].nodes).astype(np.int64)
        subgraphs[0].node_feat["term_ids"] = self.term_ids[subgraphs[0].node_feat["index"]].astype(np.int64)
        feed_dict = {}
        for i in range(self.num_layers):
            feed_dict.update(self.graph_wrappers[i].to_feed(subgraphs[i]))

        # only reindex from first subgraph
        sub_node_idx = subgraphs[0].reindex_from_parrent_nodes(batch_node)

        feed_dict["node_index"] = np.array(sub_node_idx, dtype="int64")
        feed_dict["node_real_index"] = np.array(batch_node, dtype="int64")
        feed_dict["label"] = np.array(batch_label, dtype="int64")
        return feed_dict


class BatchGraphGenerator(GraphGenerator):
    def batch_fn(self, batch_ex):
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

        if self.neg_type == "batch_neg":
            batch_neg = batch_dst
        else:
            # TODO user define shape of neg_sample
            neg_shape = batch_dst.shape
            sampled_batch_neg = alias_sample(neg_shape, self.alias, self.events)
            batch_neg = np.concatenate([batch_neg, sampled_batch_neg], 0)

        if self.phase == "train":
            # TODO user define ignore edges or not
            #ignore_edges = np.concatenate([np.stack([batch_src, batch_dst], 1), np.stack([batch_dst, batch_src], 1)], 0)
            ignore_edges = set()
        else:
            ignore_edges = set()

        nodes = np.unique(np.concatenate([batch_src, batch_dst, batch_neg], 0))
        subgraphs = graphsage_sample(self.graph, nodes, self.samples, ignore_edges=ignore_edges)
        subgraph = subgraphs[0]
        subgraphs[0].node_feat["index"] = subgraphs[0].reindex_to_parrent_nodes(subgraphs[0].nodes).astype(np.int64)
        subgraphs[0].node_feat["term_ids"] = self.term_ids[subgraphs[0].node_feat["index"]].astype(np.int64)

        # only reindex from first subgraph
        sub_src_idx = subgraphs[0].reindex_from_parrent_nodes(batch_src)
        sub_dst_idx = subgraphs[0].reindex_from_parrent_nodes(batch_dst)
        sub_neg_idx = subgraphs[0].reindex_from_parrent_nodes(batch_neg)

        user_index = np.array(sub_src_idx, dtype="int64")
        pos_item_index = np.array(sub_dst_idx, dtype="int64")
        neg_item_index = np.array(sub_neg_idx, dtype="int64")

        user_real_index = np.array(batch_src, dtype="int64")
        pos_item_real_index = np.array(batch_dst, dtype="int64")

        num_nodes = np.array([len(subgraph.nodes)], np.int32)
        num_edges = np.array([len(subgraph.edges)], np.int32)
        edges = subgraph.edges
        node_feat = subgraph.node_feat
        edge_feat = subgraph.edge_feat

        # pairwise training with label 1.
        fake_label = np.ones_like(user_index)
        
        if self.phase == "train":
            return num_nodes, num_edges, edges, node_feat["index"], node_feat["term_ids"], user_index, \
                    pos_item_index, neg_item_index, user_real_index, pos_item_real_index, fake_label
        else:
            return num_nodes, num_edges, edges, node_feat["index"], node_feat["term_ids"], user_index, \
                    pos_item_index, neg_item_index, user_real_index, pos_item_real_index

    def generator(self):
        try:
            for feed_list in super(GraphGenerator, self).generator():
                yield feed_list

        except Exception as e:
            log.exception(e)
