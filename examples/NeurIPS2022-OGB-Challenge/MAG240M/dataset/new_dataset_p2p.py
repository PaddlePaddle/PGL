# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import pdb
import argparse

import numpy as np
import yaml
from easydict import EasyDict as edict
import paddle
from paddle.io import Dataset
from pgl.graph import Graph
from ogb.lsc import MAG240MDataset
from pgl.utils.logger import log
from paddle.framework import core


class MAG240M(object):
    def __init__(self, config, ensemble_setting):
        self.num_features = 768
        self.num_classes = 153
        self.seed = config.seed
        self.data_dir = config.data_dir
        self.valid_path = config.valid_path
        self.valid_name = config.valid_name
        self.m2v_file = config.m2v_file
        self.m2v_dim = config.m2v_dim
        self.p2p_file = config.p2p_file
        self.ensemble_setting = ensemble_setting
        self.feat_mode = config.feat_mode if config.feat_mode else "cpu"

    def prepare_data(self):
        log.info("Preparing dataset...")
        dataset = MAG240MDataset(self.data_dir)

        # Get feature and graph.
        graph_file_list = []
        paper_edge_path = f'{dataset.dir}/paper_to_paper_symmetric_pgl_split'
        graph_file_list.append(paper_edge_path)
        author_edge_path = f'{dataset.dir}/paper_to_author_symmetric_pgl_split_src'
        graph_file_list.append(author_edge_path)
        author_edge_path = f'{dataset.dir}/paper_to_author_symmetric_pgl_split_dst'
        graph_file_list.append(author_edge_path)
        institution_edge_path = f'{dataset.dir}/institution_edge_symmetric_pgl_split_src'
        graph_file_list.append(institution_edge_path)
        institution_edge_path = f'{dataset.dir}/institution_edge_symmetric_pgl_split_dst'
        graph_file_list.append(institution_edge_path)

        np.random.seed(self.seed)
        valid_file = os.listdir(self.valid_path)
        valid_file = [v_n for v_n in valid_file if v_n != self.valid_name]

        if self.ensemble_setting:
            valid_temp = []
            for v_n in valid_file:
                path = os.path.join(self.valid_path, v_n)
                valid_temp.append(np.load(path))
            valid_temp = np.concatenate(valid_temp)
            self.train_idx = np.concatenate(
                [dataset.get_idx_split('train'), valid_temp])

            valid_name = os.path.join(self.valid_path, self.valid_name)
            self.val_idx = np.load(valid_name)
        else:
            self.feat_mode = "cpu"
            self.train_idx = dataset.get_idx_split('train')
            valid_name = os.path.join(self.valid_path, self.valid_name)
            self.val_idx = np.load(valid_name)

        self.test_idx = dataset.get_idx_split('test-challenge')
        self.test_dev_idx = dataset.get_idx_split('test-dev')
        N = (
            dataset.num_papers + dataset.num_authors + dataset.num_institutions
        )
        self.train_idx_mask = np.zeros([N], dtype="bool")
        if self.ensemble_setting:
            self.train_idx_mask[np.concatenate([self.train_idx, self.val_idx
                                                ])] = 1
        else:
            self.train_idx_mask[self.train_idx] = 1
        self.train_idx_mask = paddle.to_tensor(self.train_idx_mask)
        self.x = np.memmap(
            f'{dataset.dir}/full_feat.npy',
            dtype=np.float16,
            mode='r',
            shape=(N, self.num_features))
        self.id_x = np.memmap(
            f'{self.m2v_file}',
            dtype=np.float16,
            mode='r',
            shape=(N, self.m2v_dim))
        self.p2p_x = np.memmap(
            f'{self.p2p_file}',
            dtype=np.float16,
            mode='r',
            shape=(N, self.m2v_dim))
        from dist_feat import DistFeat
        self.x = DistFeat(self.x, mode=self.feat_mode)
        self.id_x = DistFeat(self.id_x, mode=self.feat_mode)
        self.p2p_x = DistFeat(self.p2p_x, mode=self.feat_mode)
        self.y = dataset.all_paper_label
        self.y = paddle.to_tensor(self.y, dtype="int64")
        year_file = f'{dataset.dir}/all_feat_year.npy'
        self.year = np.memmap(year_file, dtype=np.int32, mode='r', shape=(N, ))
        self.year = paddle.to_tensor(self.year, dtype="int32")
        self.graphs = [
            Graph.load(
                edge_path, mmap_mode='r+') for edge_path in graph_file_list
        ]
        self.prepare_csc_graph()

        def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
            def cal_angle(position, hid_idx):
                return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

            def get_posi_angle_vec(position):
                return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

            sinusoid_table = np.array(
                [get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
            sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
            sinusoid_table[:, 1::2] = np.cos(
                sinusoid_table[:, 1::2])  # dim 2i+1
            return sinusoid_table

        self.pos = get_sinusoid_encoding_table(200, 768)
        self.pos = paddle.to_tensor(self.pos)

    def prepare_csc_graph(self):
        log.info("Preparing csc graph...")
        self.csc_graphs = []
        for g in self.graphs:
            row = core.eager.to_uva_tensor(g.adj_dst_index._sorted_v, 0)
            colptr = core.eager.to_uva_tensor(g.adj_dst_index._indptr, 0)
            self.csc_graphs.append([row, colptr])

        log.info("Finish dataset")


class NodeIterDataset(Dataset):
    def __init__(self, data_index):
        self.data_index = data_index

    def __getitem__(self, idx):
        return self.data_index[idx]

    def __len__(self):
        return len(self.data_index)


class NeighborSampler(object):
    def __init__(self, csc_graphs, samples_list):
        self.csc_graphs = csc_graphs
        self.samples_list = samples_list

    def sample(self, nodes):
        graph_list = []
        for i in range(len(self.samples_list)):
            edge_split = []
            neighbors_all = []
            neighbor_counts_all = []
            for j, csc in enumerate(self.csc_graphs):
                neighbors, neighbor_counts = paddle.geometric.sample_neighbors(
                    csc[0], csc[1], nodes, sample_size=self.samples_list[i][j])
                if neighbors.shape[0] == 0:
                    edge_split.append(0)
                else:
                    edge_split.append(int(paddle.sum(neighbor_counts)))
                    neighbors_all.append(neighbors)
                    neighbor_counts_all.append(neighbor_counts)

            edge_src, edge_dst, out_nodes = \
                paddle.geometric.reindex_heter_graph(nodes, neighbors_all, neighbor_counts_all)
            edge_split = np.cumsum(edge_split).astype("int64")

            graph_list.append(
                (edge_src, edge_dst, edge_split, len(nodes), len(out_nodes)))
            nodes = out_nodes

        graph_list = graph_list[::-1]
        return graph_list, nodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument(
        "--conf", type=str, default="./configs/r_unimp_new.yaml")
    args = parser.parse_args()

    config = edict(yaml.load(open(args.conf), Loader=yaml.FullLoader))

    mag_dataset = MAG240M(config)
    mag_dataset.prepare_data()
