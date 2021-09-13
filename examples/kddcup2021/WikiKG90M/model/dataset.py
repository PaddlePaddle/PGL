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
from collections import defaultdict

import pdb
import numpy as np
from ogb.lsc import WikiKG90MDataset, WikiKG90MEvaluator


class KGDataset:
    '''Load a knowledge graph

    The folder with a knowledge graph has five files:
    * entities stores the mapping between entity Id and entity name.
    * relations stores the mapping between relation Id and relation name.
    * train stores the triples in the training set.
    * valid stores the triples in the validation set.
    * test stores the triples in the test set.

    The mapping between entity (relation) Id and entity (relation) name is stored as 'id\tname'.

    The triples are stored as 'head_name\trelation_name\ttail_name'.
    '''

    def __init__(self,
                 entity_path,
                 relation_path,
                 train_path,
                 valid_path=None,
                 test_path=None,
                 format=[0, 1, 2],
                 delimiter='\t',
                 skip_first_line=False,
                 kv_mode=False,
                 filter_mode=False):
        self.kv_mode = kv_mode
        self.delimiter = delimiter
        self.skip_first_line = skip_first_line
        self.format = format

        self.entity2id, self.n_entities = self.read_dict(entity_path)
        self.relation2id, self.n_relations = self.read_dict(relation_path)
        self.train = self.read_triple(train_path, "train")
        self.valid = self.read_triple(valid_path, "valid")
        self.test = self.read_triple(test_path, "test")
        self.head_pair = None
        self.tail_pair = None
        if filter_mode:
            self.head_pair, self.tail_pair = self.construct_true_pair()

    def construct_true_pair(self):
        head_pair = defaultdict(set)
        tail_pair = defaultdict(set)

        def add_pair(data):
            for h, r, t in zip(*data):
                head_pair[(t, r)].add(h)
                tail_pair[(h, r)].add(t)

        add_pair(self.train)
        if self.valid is not None:
            add_pair(self.valid)
        if self.test is not None:
            add_pair(self.test)
        return head_pair, tail_pair

    def read_dict(self, data_path):
        with open(data_path, 'r') as rp:
            step = 1 if self.kv_mode else -1
            items = {}
            for line in rp.readlines():
                k, v = line.strip().split(self.delimiter)[::step]
                items[k] = int(v)
        return items, len(items)

    def read_triple(self, path, mode):
        """
        mode: train/valid/test
        """
        if path is None:
            return None

        skip_first_line = self.skip_first_line
        format = self.format

        print('Reading {} triples....'.format(mode))
        heads = []
        tails = []
        rels = []
        with open(path) as f:
            if skip_first_line:
                f.readline()
            for line in f:
                triple = line.strip().split(self.delimiter)
                h, r, t = triple[format[0]], triple[format[1]], triple[format[
                    2]]
                heads.append(self.entity2id[h])
                rels.append(self.relation2id[r])
                tails.append(self.entity2id[t])

        heads = np.array(heads, dtype=np.int64)
        tails = np.array(tails, dtype=np.int64)
        rels = np.array(rels, dtype=np.int64)
        print('Finished. Read {} {} triples.'.format(len(heads), mode))
        return (heads, rels, tails)


class KGDatasetToWiki(KGDataset):
    def __init__(self, args):
        data_path = os.path.join(args.data_path, args.dataset)
        super(KGDatasetToWiki, self).__init__(
            os.path.join(data_path, 'entities.dict'),
            os.path.join(data_path, 'relations.dict'),
            os.path.join(data_path, 'train.txt'),
            os.path.join(data_path, 'valid.txt'),
            os.path.join(data_path, 'test.txt'),
            filter_mode=args.filter)
        self.valid = self.get_percent(self.valid, args.eval_percent)
        self.test = self.get_percent(self.test, args.test_percent)
        self.valid_dict = self.convert_to_wiki_dict(self.valid)
        self.test_dict = self.convert_to_wiki_dict(self.test)
        self.train = np.stack(self.train)
        self.valid = None
        self.test = None
        self.entity_feat = None
        self.relation_feat = None

    def get_percent(self, data, percent):
        if percent != 1:
            num_percent = int(len(data[0]) * percent)
            data = tuple([x[:num_percent] for x in data])
        return data

    def convert_to_wiki_dict(self, data):
        data_dict = {
            'h,r->t': {
                'hr': np.stack(data[:2]).T,
                't_candidate': np.tile(
                    np.arange(self.n_entities).reshape(1, -1),
                    (len(data[2]), 1)),
                't_correct_index': data[2]
            }
        }
        return data_dict

    def get_random_partition(self, k):
        '''k: number of partitions'''

    @property
    def emap_fname(self):
        return None

    @property
    def rmap_fname(self):
        return None


class KGDatasetWiki(KGDataset):
    '''Load a knowledge graph wikikg
    '''

    def __init__(self, args, path, name='wikikg90m'):
        self.name = name
        self.dataset = WikiKG90MDataset(path)
        if args.eval_percent != 1.:
            num_valid = len(self.dataset.valid_dict['h,r->t']['hr'])
            num_valid = int(num_valid * args.eval_percent)
            self.dataset.valid_dict['h,r->t']['hr'] = self.dataset.valid_dict[
                'h,r->t']['hr'][:num_valid]
            self.dataset.valid_dict['h,r->t'][
                't_candidate'] = self.dataset.valid_dict['h,r->t'][
                    't_candidate'][:num_valid]
            self.dataset.valid_dict['h,r->t'][
                't_correct_index'] = self.dataset.valid_dict['h,r->t'][
                    't_correct_index'][:num_valid]
            print("num_valid", num_valid)
        if args.test_percent != 1.:
            num_test = len(self.dataset.test_dict['h,r->t']['hr'])
            num_test = int(num_test * args.test_percent)
            self.dataset.test_dict['h,r->t']['hr'] = self.dataset.test_dict[
                'h,r->t']['hr'][:num_test]
            self.dataset.test_dict['h,r->t'][
                't_candidate'] = self.dataset.test_dict['h,r->t'][
                    't_candidate'][:num_test]
            print("num_test", num_test)
        if args.train_percent != 1.:
            num_train = self.dataset.train_hrt.shape[0]
            num_train = int(num_train * args.train_percent)
            print("num_train", num_train)
            self.train = self.dataset.train_hrt.T[:, :num_train]
        else:
            self.train = self.dataset.train_hrt.T

        self.n_entities = self.dataset.num_entities
        self.n_relations = self.dataset.num_relations
        self.valid = None
        self.test = None
        self.valid_dict = self.dataset.valid_dict
        self.test_dict = self.dataset.test_dict
        self.entity_feat = self.dataset.entity_feat
        self.relation_feat = self.dataset.relation_feat
        if 't,r->h' in self.valid_dict:
            del self.valid_dict['t,r->h']
        if 't,r->h' in self.test_dict:
            del self.valid_dict['t,r->h']

    @property
    def emap_fname(self):
        return None

    @property
    def rmap_fname(self):
        return None


class KGDatasetCN31(KGDataset):
    def __init__(self, path, name='toy'):
        self.name = name
        # self.dataset = WikiKG90MDataset(path)
        self.train = np.fromfile(
            os.path.join(path, "train_triple_id.txt"),
            sep="\t").astype("int32").reshape([-1, 3]).T
        self.valid = np.fromfile(
            os.path.join(path, "valid_triple_id.txt"),
            sep="\t").astype("int32").reshape([-1, 3]).T
        self.test = np.fromfile(
            os.path.join(path, "test_triple_id.txt"),
            sep="\t").astype("int32").reshape([-1, 3]).T

        self.n_entities = self.train.max() + 1
        self.n_relations = self.train[1, :].max() + 1
        self.feat_dim = 768
        self.entity_feat = np.random.randn(
            self.n_entities,
            self.feat_dim)  #np.load(os.path.join(path, "entity_feat.npy"))
        self.relation_feat = np.random.randn(
            self.n_relations,
            self.feat_dim)  #np.load(os.path.join(path, "relation_feat.npy"))
        self.entity_degree = np.ones((self.n_entities, 2))
        self.valid_dict = {
            'h,r->t': {
                'hr': np.random.randint(
                    15, size=(50, 2)),
                't_candidate': np.random.randint(
                    15, size=(50, 10)),
                't_correct_index': np.random.randint(
                    5, size=(50, ))
            }
        }
        self.test_dict = {
            'h,r->t': {
                'hr': np.random.randint(
                    15, size=(50, 2)),
                't_candidate': np.random.randint(
                    15, size=(50, 10)),
                't_correct_index': np.random.randint(
                    5, size=(50, ))
            }
        }

    @property
    def emap_fname(self):
        return None

    @property
    def rmap_fname(self):
        return None


class KGDatasetToy(KGDataset):
    def __init__(self, path, name='toy'):
        self.name = name
        # self.dataset = WikiKG90MDataset(path)
        self.n_entities = 10000
        self.n_relations = 15
        self.feat_dim = 768
        self.train = np.random.randint(
            15,
            size=(3, 100000))  # np.load(os.path.join(path, "train_hrt.npy")).T
        self.valid = None
        self.test = None
        self.entity_feat = np.random.randn(
            self.n_entities,
            self.feat_dim)  #np.load(os.path.join(path, "entity_feat.npy"))
        self.relation_feat = np.random.randn(
            self.n_relations,
            self.feat_dim)  #np.load(os.path.join(path, "relation_feat.npy"))
        self.entity_degree = np.ones((self.n_entities, 2))
        self.valid_dict = {
            'h,r->t': {
                'hr': np.random.randint(
                    15, size=(50, 2)),
                't_candidate': np.random.randint(
                    15, size=(50, 10)),
                't_correct_index': np.random.randint(
                    5, size=(50, ))
            }
        }
        self.test_dict = {
            'h,r->t': {
                'hr': np.random.randint(
                    15, size=(50, 2)),
                't_candidate': np.random.randint(
                    15, size=(50, 10)),
                't_correct_index': np.random.randint(
                    5, size=(50, ))
            }
        }

    @property
    def emap_fname(self):
        return None

    @property
    def rmap_fname(self):
        return None


def get_dataset(args,
                data_path,
                data_name,
                format_str='built_in',
                delimiter='\t',
                files=None,
                has_edge_importance=False):
    if format_str == 'built_in':
        if data_name == "wikikg90m":
            dataset = KGDatasetWiki(args=args, path=data_path)
        elif data_name == "toy":
            dataset = KGDatasetToy(data_path)
        elif data_name == "cn31":
            dataset = KGDatasetCN31(data_path)
        elif data_name in ["FB15k-237", "FB15k", "wikikg2"]:
            dataset = KGDatasetToWiki(args=args)
        else:
            assert False, "Unknown dataset {}".format(data_name)
    else:
        dataset = KGDatasetToy(data_path, args=args)

    return dataset
