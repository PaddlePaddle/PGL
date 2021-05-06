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

import os
import sys
import yaml
import pgl
import time
import copy
import numpy as np
import os.path as osp
from pgl.utils.logger import log
from pgl.graph import Graph
from pgl import graph_kernel
from pgl.sampling.custom import subgraph
from ogb.lsc import MAG240MDataset, MAG240MEvaluator
from dataset.base_dataset import BaseDataGenerator
import time
from tqdm import tqdm


class MAG240M(object):
    """Iterator"""

    def __init__(self, data_dir, seed=123):
        self.data_dir = data_dir
        self.num_features = 768
        self.num_classes = 153
        self.seed = seed

    def prepare_data(self):
        dataset = MAG240MDataset(self.data_dir)

        paper_edge_path = f'{dataset.dir}/paper_to_paper_symmetric_pgl'
        t = time.perf_counter()
        if not osp.exists(paper_edge_path):
            log.info('Converting adjacency matrix...')
            edge_index = dataset.edge_index('paper', 'cites', 'paper')
            edge_index = edge_index.T

            edges_new = np.zeros((edge_index.shape[0], 2))
            edges_new[:, 0] = edge_index[:, 1]
            edges_new[:, 1] = edge_index[:, 0]

            edge_index = np.vstack((edge_index, edges_new))
            #            edge_index = np.unique(edge_index, axis=0)

            graph = Graph(edge_index)
            graph.adj_dst_index
            graph.dump(paper_edge_path)
            log.info(f'Done! [{time.perf_counter() - t:.2f}s]')

        edge_path = f'{dataset.dir}/full_edge_symmetric_pgl'
        t = time.perf_counter()
        if not osp.exists(edge_path):
            log.info('Converting adjacency matrix...')

            # paper
            log.info('adding paper edges')
            paper_graph = Graph.load(paper_edge_path, mmap_mode='r+')
            rows, cols = [paper_graph.edges[:, 0]], [paper_graph.edges[:, 1]]

            # author
            log.info('adding author edges')
            edge_index = dataset.edge_index('author', 'writes', 'paper')
            edge_index = edge_index.T
            row, col = edge_index[:, 0], edge_index[:, 1]
            row += dataset.num_papers
            rows += [row, col]
            cols += [col, row]

            # institution
            log.info('adding institution edges')
            edge_index = dataset.edge_index('author', 'institution')
            edge_index = edge_index.T
            row, col = edge_index[:, 0], edge_index[:, 1]
            row += dataset.num_papers
            col += dataset.num_papers + dataset.num_authors
            rows += [row, col]
            cols += [col, row]

            # edge_type
            log.info('building edge type')
            edge_types = [
                np.full(
                    x.shape, i, dtype='int32') for i, x in enumerate(rows)
            ]
            edge_types = np.concatenate(edge_types, axis=0)

            log.info('building edges')
            row = np.concatenate(rows, axis=0)
            del rows

            col = np.concatenate(cols, axis=0)
            del cols

            edge_index = np.stack([row, col], axis=1)
            N = dataset.num_papers + dataset.num_authors + dataset.num_institutions
            full_graph = Graph(
                edge_index, num_nodes=N, edge_feat={'edge_type': edge_types})
            full_graph.adj_dst_index
            full_graph.dump(edge_path)
            log.info(
                f'Done! finish full_edge [{time.perf_counter() - t:.2f}s]')

        path = f'{dataset.dir}/full_feat.npy'

        author_feat_path = f'{dataset.dir}/author_feat.npy'

        institution_feat_path = f'{dataset.dir}/institution_feat.npy'

        t = time.perf_counter()
        if not osp.exists(path):  # Will take ~3 hours...
            print('Generating full feature matrix...')

            node_chunk_size = 100000
            N = (dataset.num_papers + dataset.num_authors +
                 dataset.num_institutions)

            paper_feat = dataset.paper_feat

            author_feat = np.memmap(
                author_feat_path,
                dtype=np.float16,
                shape=(dataset.num_authors, self.num_features),
                mode='r')

            institution_feat = np.memmap(
                institution_feat_path,
                dtype=np.float16,
                shape=(dataset.num_institutions, self.num_features),
                mode='r')

            x = np.memmap(
                path,
                dtype=np.float16,
                mode='w+',
                shape=(N, self.num_features))

            print('Copying paper features...')
            start_idx = 0
            end_idx = dataset.num_papers
            for i in tqdm(range(start_idx, end_idx, node_chunk_size)):
                j = min(i + node_chunk_size, end_idx)
                x[i:j] = paper_feat[i:j]
            del paper_feat

            print('Copying author feature...')
            start_idx = dataset.num_papers
            end_idx = dataset.num_papers + dataset.num_authors
            for i in tqdm(range(start_idx, end_idx, node_chunk_size)):
                j = min(i + node_chunk_size, end_idx)
                x[i:j] = author_feat[i - start_idx:j - start_idx]
            del author_feat

            print('Copying institution feature...')
            start_idx = dataset.num_papers + dataset.num_authors
            end_idx = dataset.num_papers + dataset.num_authors + dataset.num_institutions
            for i in tqdm(range(start_idx, end_idx, node_chunk_size)):
                j = min(i + node_chunk_size, end_idx)
                x[i:j] = institution_feat[i - start_idx:j - start_idx]
            del institution_feat

            x.flush()
            del x
            print(f'Done! [{time.perf_counter() - t:.2f}s]')

        np.random.seed(self.seed)
        self.train_idx = dataset.get_idx_split('train')
        np.random.shuffle(self.train_idx)

        self.val_idx = dataset.get_idx_split('valid')
        self.test_idx = dataset.get_idx_split('test')

        N = dataset.num_papers + dataset.num_authors + dataset.num_institutions
        self.x = np.memmap(
            f'{dataset.dir}/full_feat.npy',
            dtype=np.float16,
            mode='r',
            shape=(N, self.num_features))

        self.y = dataset.all_paper_label

        self.graph = Graph.load(edge_path, mmap_mode='r+')
        self.graph._edge_feat['edge_type'] = self.graph._edge_feat[
            'edge_type'].astype('int32')

        log.info(f'Done! [{time.perf_counter() - t:.2f}s]')

    @property
    def train_examples(self, ):
        # Filters
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        trainer_num = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
        count_line = 0

        #np.random.shuffle(self.train_idx)
        for idx in self.train_idx:
            count_line += 1
            if count_line % trainer_num == trainer_id:
                yield idx

    @property
    def eval_examples(self, ):
        for idx in self.val_idx:
            yield idx

    @property
    def test_examples(self, ):
        for idx in self.test_idx:
            yield idx


def add_self_loop(graph, sub_nodes=None):
    '''add_self_loop_for_subgraph
    '''
    assert not graph.is_tensor(), "You must call Graph.numpy() first."

    if sub_nodes is not None:
        self_loop_edges = np.zeros((sub_nodes.shape[0], 2))
        self_loop_edges[:, 0] = self_loop_edges[:, 1] = sub_nodes
    else:
        self_loop_edges = np.zeros((graph.num_nodes, 2))
        self_loop_edges[:, 0] = self_loop_edges[:, 1] = np.arange(
            graph.num_nodes)
    edges = np.vstack((graph.edges, self_loop_edges))
    edges = np.unique(edges, axis=0)
    new_g = Graph(
        edges=edges,
        num_nodes=graph.num_nodes,
        node_feat=graph.node_feat,
        edge_feat=graph.edge_feat)
    return new_g


def traverse(item):
    """traverse
    """
    if isinstance(item, list) or isinstance(item, np.ndarray):
        for i in iter(item):
            for j in traverse(i):
                yield j
    else:
        yield item


def flat_node_and_edge(nodes):
    """flat_node_and_edge
    """
    nodes = list(set(traverse(nodes)))
    return nodes


def neighbor_sample(graph, nodes, samples):
    assert not graph.is_tensor(), "You must call Graph.numpy() first."

    graph_list = []
    for max_deg in samples:
        start_nodes = copy.deepcopy(nodes)
        edges = []
        edge_ids = []
        if max_deg == -1:
            pred_nodes, pred_eids = graph.predecessor(
                start_nodes, return_eids=True)
        else:
            pred_nodes, pred_eids = graph.sample_predecessor(
                start_nodes, max_degree=max_deg, return_eids=True)

        for dst_node, src_nodes, src_eids in zip(start_nodes, pred_nodes,
                                                 pred_eids):
            for src_node, src_eid in zip(src_nodes, src_eids):
                edges.append((src_node, dst_node))
                edge_ids.append(src_eid)

        neigh_nodes = [start_nodes, pred_nodes]
        neigh_nodes = flat_node_and_edge(neigh_nodes)

        from_reindex = {x: i for i, x in enumerate(neigh_nodes)}
        sub_node_index = graph_kernel.map_nodes(nodes, from_reindex)

        sg = subgraph(
            graph,
            eid=edge_ids,
            nodes=neigh_nodes,
            edges=edges,
            with_node_feat=False,
            with_edge_feat=True)
        #         sg = add_self_loop(sg, sub_node_index)

        graph_list.append((sg, neigh_nodes, sub_node_index))
        nodes = neigh_nodes

    graph_list = graph_list[::-1]
    return graph_list


class DataGenerator(BaseDataGenerator):
    def __init__(self, dataset, samples, batch_size, num_workers, data_type):

        super(DataGenerator, self).__init__(
            buf_size=1000,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True if data_type == 'train' else False)

        self.dataset = dataset
        self.samples = samples
        if data_type == 'train':
            self.line_examples = self.dataset.train_examples
        elif data_type == 'eval':
            self.line_examples = self.dataset.eval_examples
        else:
            self.line_examples = self.dataset.test_examples

    def batch_fn(self, batch_nodes):

        graph_list = neighbor_sample(self.dataset.graph, batch_nodes,
                                     self.samples)

        neigh_nodes = graph_list[0][1]
        y = self.dataset.y[batch_nodes]
        return graph_list, neigh_nodes, y

    def post_fn(self, batch):
        graph_list, neigh_nodes, y = batch
        x = self.dataset.x[neigh_nodes]
        return graph_list, x, y


if __name__ == "__main__":
    root = sys.argv[1]
    print(root)
    dataset = MAG240M(root)
    dataset.prepare_data()
