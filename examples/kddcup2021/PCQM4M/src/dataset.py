# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
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
sys.path.append("../")
import json
import glob
import copy
import time
import tqdm
import argparse
import numpy as np
import pickle as pkl
from collections import OrderedDict, namedtuple

from ogb.lsc import PCQM4MDataset, PCQM4MEvaluator
#  from ogb.utils import smiles2graph
from rdkit.Chem import AllChem

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import pgl
from pgl.utils.data.dataset import Dataset, StreamDataset, HadoopDataset
from pgl.utils.data import Dataloader
from pgl.utils.logger import log

from utils.config import prepare_config, make_dir

def load_vocab(vocab_file, freq=0):
    vocab = {"": 0, "CUT": 1}
    with open(vocab_file, 'r') as f:
        for line in f:
            fields = line.rstrip("\r\n").split('\t')
            if int(fields[1]) > freq:
                vocab[fields[0]] = len(vocab)
            else:
                vocab[fields[0]] = vocab["CUT"]

    return vocab

def getmorganfingerprint(mol):
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, 2))

def getmaccsfingerprint(mol):
    fp = AllChem.GetMACCSKeysFingerprint(mol)
    return [int(b) for b in fp.ToBitString()]

class Subset(Dataset):
    r"""
    Subset of a dataset at specified indices.
    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices, mode='train'):
        self.dataset = dataset
        if paddle.distributed.get_world_size() == 1 or mode != "train":
            self.indices = indices
        else:
            self.indices = indices[int(paddle.distributed.get_rank())::int(
                paddle.distributed.get_world_size())]
        self.mode = mode

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

class MolDataset(Dataset):
    def __init__(self, config, mode="train"):
        log.info("dataset_type is %s" % self.__class__.__name__)
        self.config = config
        self.mode = mode
        self.transform = config.transform
        self.raw_dataset = PCQM4MDataset(config.base_data_path, only_smiles=True)

        self.graph_list = None
        if not config.debug and self.config.preprocess_file is not None:
            log.info("preprocess graph data in %s" % self.__class__.__name__)
            processed_path = os.path.join(self.config.base_data_path, "pgl_processed")
            if not os.path.exists(processed_path):
                os.makedirs(processed_path)
            data_file = os.path.join(processed_path, self.config.preprocess_file)

            if os.path.exists(data_file):
                log.info("loading graph data from pkl file")
                self.graph_list = pkl.load(open(data_file, "rb"))
            else:
                log.info("loading graph data from smiles data using %s transform" \
                        % self.transform)
                self.graph_list = []
                for i in tqdm.tqdm(range(len(self.raw_dataset))):
                    # num_nodes, edge_index, node_feat, edge_feat, label
                    smiles, label = self.raw_dataset[i]
                    g = getattr(self, self.transform)(smiles, label)
                    self.graph_list.append(g)

                pkl.dump(self.graph_list, open(data_file, 'wb'))
        else:
            processed_path = os.path.join(self.config.base_data_path, "pgl_processed")
            vocab_file = os.path.join(processed_path, "junc_vocab.txt")
            self.vocab = load_vocab(vocab_file)

    def get_idx_split(self):
        if self.config.debug:
            split_idx = {'train': [i for i in range(800)],
                    'valid': [i + 800 for i in range(100)],
                    'test': [i + 800 for i in range(100)]}
            return split_idx
        else:
            return self.raw_dataset.get_idx_split()

    def __getitem__(self, idx):
        return self.graph_list[idx]

    def __len__(self):
        return len(self.raw_dataset)

class ExMolDataset(Dataset):
    def __init__(self, config, mode='train', transform=None):
        self.config = config
        self.mode = mode
        self.transform = transform
        self.raw_dataset = PCQM4MDataset(
            config.base_data_path, only_smiles=True)

        log.info("preprocess graph data in %s" % self.__class__.__name__)

        graph_path = os.path.join(self.config.preprocess_file, "mmap_graph")
        label_file = os.path.join(self.config.preprocess_file, "label.npy")

        self.graph = pgl.Graph.load(graph_path)
        self.label = np.load(label_file)

    def get_idx_split(self):
        if self.config.debug:
            split_idx = {'train': [i for i in range(800)],
                    'valid': [i + 800 for i in range(100)],
                    'test': [i + 900 for i in range(100)]}
            return split_idx
        else:
            return self.raw_dataset.get_idx_split()

    def get_cross_idx_split(self):
        if self.config.debug:
            split_idx = {'cross_train_1': [i for i in range(800)],
                    'cross_train_2': [i for i in range(800)],
                    'cross_valid_1': [i + 800 for i in range(100)],
                    'cross_valid_2': [i + 800 for i in range(100)],
                    'valid_left_1percent': [i + 800 for i in range(100)],
                    'test': [i + 900 for i in range(100)]}
            return split_idx
        else:
            cross_split_idx_file = os.path.join(self.config.base_data_path, "cross_split.pkl")
            split_idx = pkl.load(open(cross_split_idx_file, 'rb'))
            return split_idx

    def __getitem__(self, idx):
        num_nodes = self.graph._graph_node_index[idx + 1] - self.graph._graph_node_index[idx]
        node_shift = self.graph._graph_node_index[idx]
        edges = self.graph.edges[self.graph._graph_edge_index[idx]:self.graph._graph_edge_index[idx + 1]]
        edges = edges - node_shift
        edge_feat = {}
        for key, value in self.graph.edge_feat.items():
            edge_feat[key] = value[self.graph._graph_edge_index[idx]:self.graph._graph_edge_index[idx + 1]]
        node_feat = {}
        for key, value in self.graph.node_feat.items():
            node_feat[key] = value[self.graph._graph_node_index[idx]:self.graph._graph_node_index[idx + 1]]

        smiles, label = self.raw_dataset[idx]
        return (pgl.Graph(num_nodes=num_nodes, edges=edges, node_feat=node_feat, edge_feat=edge_feat), self.label[idx], smiles)

    def __len__(self):
        return self.graph.num_graph


class AuxDataset(Dataset):
    def __init__(self, config, mode='train', transform=None):
        self.config = config
        self.mode = mode
        self.transform = transform
        self.raw_dataset = PCQM4MDataset(
            config.base_data_path, only_smiles=True)

        log.info("preprocess graph data in %s" % self.__class__.__name__)

        graph_path = os.path.join(self.config.preprocess_file, "mmap_graph")
        label_file = os.path.join(self.config.preprocess_file, "label.npy")

        self.graph = pgl.Graph.load(graph_path)
        self.pretrain_info_list = pkl.load(open(config.pretrian_path,"rb"))
        print(f"len of pretrain data: {len(self.pretrain_info_list)}")
        self.label = np.load(label_file)

    def get_idx_split(self):
        return self.raw_dataset.get_idx_split()

    def get_cross_idx_split(self):
        cross_split_idx_file = os.path.join(self.config.base_data_path, "cross_split.pkl")
        split_idx = pkl.load(open(cross_split_idx_file, 'rb'))
        return split_idx

    def __getitem__(self, idx):
        num_nodes = self.graph._graph_node_index[idx + 1] - self.graph._graph_node_index[idx]        
        node_shift = self.graph._graph_node_index[idx]
        edges = self.graph.edges[self.graph._graph_edge_index[idx]:self.graph._graph_edge_index[idx + 1]]
        edges = edges - node_shift
        edge_feat = {}
        for key, value in self.graph.edge_feat.items():
            edge_feat[key] = value[self.graph._graph_edge_index[idx]:self.graph._graph_edge_index[idx + 1]]
        node_feat = {}
        for key, value in self.graph.node_feat.items():
            node_feat[key] = value[self.graph._graph_node_index[idx]:self.graph._graph_node_index[idx + 1]]
        #pretrain information
        pretrain_info = {}
        cid = self.pretrain_info_list[idx]["context_id"]
        edge_index = self.pretrain_info_list[idx]["edge_index"]
        tid = self.pretrain_info_list[idx]["twohop_context"]
        if num_nodes!=len(tid):
            print(f"idx {idx} num_nodes is : {num_nodes} and len of tid is : {len(tid)}, they are not equal")
            exit(0)
        bond_angle_index = self.pretrain_info_list[idx]["bond_angle_index"]
        bond_angle = self.pretrain_info_list[idx]["bond_angle"]
        dft_success = self.pretrain_info_list[idx]["dft_success"]
        bond_angle_mask = np.array(self.pretrain_info_list[idx]["bond_angle"] * 0 + dft_success, dtype=bool)
        edge_attr_float = np.array(self.pretrain_info_list[idx]["edge_feat_float"])
        edge_attr_float_mask = np.array(self.pretrain_info_list[idx]["edge_feat_float"].reshape(-1) * 0 + dft_success, dtype=bool)
        pretrain_info["edge_index"] = np.array(edge_index)
        pretrain_info["tid"] = np.array(tid, dtype=int)
        pretrain_info["bond_angle_index"] = bond_angle_index
        pretrain_info["bond_angle"] = bond_angle
        pretrain_info["bond_angle_mask"] = bond_angle_mask
        pretrain_info["edge_attr_float" ] = edge_attr_float
        pretrain_info["edge_attr_float_mask"] = edge_attr_float_mask
        smiles, label = self.raw_dataset[idx]
        return (pgl.Graph(num_nodes=num_nodes, edges=edges, node_feat=node_feat, edge_feat=edge_feat), self.label[idx], smiles, pretrain_info)

    def __len__(self):
        return self.graph.num_graph
    
class CollateFn(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, batch_data):
        fn = getattr(self, self.config.collate_type)
        return fn(batch_data)

    def new_graph_collatefn(self, batch_data):
        # for graph_data_additional_features_0424.pkl
        # with graph_transform in mol_features_extract.py
        graph_list = []
        labels = []
        for gdata in batch_data:
            efeat = np.delete(gdata['edge_feat'], -1, axis=1) # remove 3d dist
            g = pgl.Graph(edges=gdata['edge_index'].T,
                    num_nodes=gdata['num_nodes'],
                    node_feat={'feat': gdata['node_feat']},
                    edge_feat={'feat': efeat})
            graph_list.append(g)
            labels.append(gdata['label'])

        labels = np.array(labels, dtype="float32")
        g = pgl.Graph.batch(graph_list)

        return {'graph': g}, labels

    def graph_collatefn(self, batch_data):
        graph_list = []
        labels = []
        for gdata in batch_data:
            g = pgl.Graph(edges=gdata['edge_index'].T,
                    num_nodes=gdata['num_nodes'],
                    node_feat={'feat': gdata['node_feat']},
                    edge_feat={'feat': gdata['edge_feat']})
            graph_list.append(g)
            labels.append(gdata['label'])

        labels = np.array(labels, dtype="float32")
        g = pgl.Graph.batch(graph_list)

        return {'graph': g}, labels

    def quality_graph_collatefn(self, batch_data):
        graph_list = []
        labels = []
        for gdata in batch_data:
            g = pgl.Graph(edges=gdata['mol_graph']['edge_index'].T,
                    num_nodes=gdata['mol_graph']['num_nodes'],
                    node_feat={'feat': gdata['mol_graph']['node_feat']},
                    edge_feat={'feat': gdata['mol_graph']['edge_feat']})
            graph_list.append(g)
            labels.append(gdata['label'])

        labels = np.array(labels, dtype="float32")
        g = pgl.Graph.batch(graph_list)

        return {'graph': g}, labels


    def junc_collatefn(self, batch_data):
        graph_list = []
        labels = []
        junc_graph_list = []
        mol2junc_list = []

        g_offset = 0
        junc_g_offset = 0
        for gdata in batch_data:
            g = pgl.Graph(edges=gdata['mol_graph']['edge_index'].T,
                    num_nodes=gdata['mol_graph']['num_nodes'],
                    node_feat={'feat': gdata['mol_graph']['node_feat']},
                    edge_feat={'feat': gdata['mol_graph']['edge_feat']})

            num_nodes = gdata['junction_tree']['num_nodes']
            if num_nodes > 0:
                nfeat = np.array(gdata['junction_tree']['junc_dict'], 
                        dtype="int64").reshape(-1, 1)
                junc_g = pgl.Graph(edges=gdata['junction_tree']['edge_index'].T,
                        num_nodes=num_nodes,
                        node_feat={'feat': nfeat})

                offset = np.array([g_offset, junc_g_offset], dtype="int64")

                mol2junc = gdata['mol2juct'] + offset
                junc_g_offset += junc_g.num_nodes

                junc_graph_list.append(junc_g)
                mol2junc_list.append(mol2junc)

            graph_list.append(g)
            labels.append(gdata['label'])
            g_offset += g.num_nodes

        mol2junc = np.concatenate(mol2junc_list, axis=0)

        labels = np.array(labels, dtype="float32")
        g = pgl.Graph.batch(graph_list)
        junc_g = pgl.Graph.batch(junc_graph_list)

        return {'graph': g, 'junc_graph': junc_g, 'mol2junc': mol2junc}, labels

    def coord3_junc_collatefn(self, batch_data):
        graph_list = []
        labels = []
        junc_graph_list = []
        mol2junc_list = []

        g_offset = 0
        junc_g_offset = 0
        for gdata in batch_data:
            g = pgl.Graph(edges=gdata['mol_graph']['edge_index'].T,
                    num_nodes=gdata['mol_graph']['num_nodes'],
                    node_feat={'feat': gdata['mol_graph']['node_feat'],
                        '3d': gdata['mol_coord']},
                    edge_feat={'feat': gdata['mol_graph']['edge_feat']})

            num_nodes = gdata['junction_tree']['num_nodes']
            if num_nodes > 0:
                nfeat = np.array(gdata['junction_tree']['junc_dict'], 
                        dtype="int64").reshape(-1, 1)
                junc_g = pgl.Graph(edges=gdata['junction_tree']['edge_index'].T,
                        num_nodes=num_nodes,
                        node_feat={'feat': nfeat})

                offset = np.array([g_offset, junc_g_offset], dtype="int64")

                mol2junc = gdata['mol2juct'] + offset
                junc_g_offset += junc_g.num_nodes

                junc_graph_list.append(junc_g)
                mol2junc_list.append(mol2junc)

            graph_list.append(g)
            labels.append(gdata['label'])
            g_offset += g.num_nodes

        mol2junc = np.concatenate(mol2junc_list, axis=0)

        labels = np.array(labels, dtype="float32")
        g = pgl.Graph.batch(graph_list)
        junc_g = pgl.Graph.batch(junc_graph_list)

        return {'graph': g, 'junc_graph': junc_g, 'mol2junc': mol2junc}, labels

    def fp_collatefn(self, batch_data):
        graph_list = []
        labels = []
        mgf_list = []
        maccs_list = []
        for gdata in batch_data:
            g = pgl.Graph(edges=gdata['edge_index'].T,
                    num_nodes=gdata['num_nodes'],
                    node_feat={'feat': gdata['node_feat']},
                    edge_feat={'feat': gdata['edge_feat']})
            graph_list.append(g)
            labels.append(gdata['label'])
            mgf_list.append(gdata['mgf'])
            maccs_list.append(gdata['maccs'])

        labels = np.array(labels, dtype="float32")
        g = pgl.Graph.batch(graph_list)
        mgf_feat = np.array(mgf_list, dtype="float32")
        maccs_feat = np.array(maccs_list, dtype="float32")

        others = {}
        others['mgf'] = mgf_feat
        others['maccs'] = maccs_feat

        return {'graph': g, 'mgf': mgf_feat, 'maccs': maccs_feat}, labels

    def ex_collatefn(self, batch_data):
        graph_list = []
        #  full_graph_list = []
        labels = []
        smiles_list = []
        #for gdata in batch_data:
        for g, l, s in batch_data:
            graph_list.append(g)
            #  full_g = pgl.Graph(num_nodes=g.num_nodes, edges=make_full(g.num_nodes))
            #  full_graph_list.append(full_g)
            labels.append(l)
            smiles_list.append(s)

        labels = np.array(labels, dtype="float32")
        g = pgl.Graph.batch(graph_list)
        #  full_g = pgl.Graph.batch(full_graph_list)
        #full_g = None
        others = {'smiles': smiles_list}
        return {'graph': g}, labels, others
    
    def aux_collatefn(self, batch_data):
        graph_list = []
        #  full_graph_list = []
        labels = []
        smiles_list = []
        pretrain_info_list = []
        tid_list = []
        edge_index_list = []
        bond_angle_list = []
        bond_angle_index_list = []
        bond_angle_mask_list = []
        edge_attr_float_list = []
        edge_attr_float_mask_list = []
        #for gdata in batch_data:
        total_node_num = 0
        for g, l, s , pretrain_info in batch_data:
            graph_list.append(g)
            
            #  full_g = pgl.Graph(num_nodes=g.num_nodes, edges=make_full(g.num_nodes))
            #  full_graph_list.append(full_g)
            labels.append(l)
            smiles_list.append(s)
            tid_list.append(pretrain_info["tid"].reshape(-1,1))
            edge_index_list.append(pretrain_info["edge_index"])
            bond_angle_list.append(pretrain_info["bond_angle"].reshape(-1,1))
            bond_angle_index_list.append(pretrain_info["bond_angle_index"] + total_node_num)
            bond_angle_mask_list.append(pretrain_info["bond_angle_mask"].reshape(-1,1))
            edge_attr_float_list.append(pretrain_info["edge_attr_float"].reshape(-1,1))
            edge_attr_float_mask_list.append(pretrain_info["edge_attr_float_mask"].reshape(-1,1))
            total_node_num += g.num_nodes
                                           

        tid_list = np.concatenate(tid_list)
        edge_index_list = np.concatenate(edge_index_list, axis=1)
        bond_angle_list = np.concatenate(bond_angle_list).astype('float32')
        bond_angle_index_list = np.concatenate(bond_angle_index_list, axis=1)
        bond_angle_mask_list = np.concatenate(bond_angle_mask_list)
        edge_attr_float_list = np.concatenate(edge_attr_float_list).astype('float32')
        edge_attr_float_mask_list = np.concatenate(edge_attr_float_mask_list)
        labels = np.array(labels, dtype="float32")
        g = pgl.Graph.batch(graph_list)
        others = {'smiles': smiles_list}
        return {'graph': g, "tid": tid_list, "edge_index":edge_index_list, "bond_angle":bond_angle_list, "bond_angle_mask": bond_angle_mask_list, "bond_angle_index": bond_angle_index_list, "edge_attr_float":edge_attr_float_list, "edge_attr_float_mask":edge_attr_float_mask_list}, labels, others

def test_dataset(config):
    print("loading dataset")
    ds = MolDataset(config)
    split_idx = ds.get_idx_split()
    train_ds = Subset(ds, split_idx['train'], mode='train')
    valid_ds = Subset(ds, split_idx['valid'], mode='valid')
    test_ds = Subset(ds, split_idx['test'], mode='test')
    print("Train exapmles: %s" % len(train_ds))
    print("Valid exapmles: %s" % len(valid_ds))
    print("Test exapmles: %s" % len(test_ds))

    for i in range(len(train_ds)):
        gdata = train_ds[i]
        print("nfeat: ", np.sum(gdata['node_feat']))
        print("edges: ", np.sum(gdata['edge_index']))
        print("label: ", gdata['label'])
        if i == 10:
            break

    print("valid data")
    for i in range(len(valid_ds)):
        gdata = valid_ds[i]
        print("nfeat: ", np.sum(gdata['node_feat']))
        print("edges: ", np.sum(gdata['edge_index']))
        print("label: ", gdata['label'])
        if i == 10:
            break

if __name__=="__main__":
    config = prepare_config("./config.yaml", isCreate=False, isSave=False)
    test_dataset(config)
