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

import math
import os, sys
import numpy as np
import pdb
from ogb.lsc import WikiKG90MDataset, WikiKG90MEvaluator
import pickle
import paddle
from paddle.io import Dataset, DataLoader


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
                 skip_first_line=False):
        self.delimiter = delimiter
        self.entity2id, self.n_entities = self.read_entity(entity_path)
        self.relation2id, self.n_relations = self.read_relation(relation_path)
        self.train = self.read_triple(train_path, "train", skip_first_line,
                                      format)
        if valid_path is not None:
            self.valid = self.read_triple(valid_path, "valid", skip_first_line,
                                          format)
        else:
            self.valid = None
        if test_path is not None:
            self.test = self.read_triple(test_path, "test", skip_first_line,
                                         format)
        else:
            self.test = None

    def read_entity(self, entity_path):
        with open(entity_path) as f:
            entity2id = {}
            for line in f:
                eid, entity = line.strip().split(self.delimiter)
                entity2id[entity] = int(eid)

        return entity2id, len(entity2id)

    def read_relation(self, relation_path):
        with open(relation_path) as f:
            relation2id = {}
            for line in f:
                rid, relation = line.strip().split(self.delimiter)
                relation2id[relation] = int(rid)

        return relation2id, len(relation2id)

    def read_triple(self, path, mode, skip_first_line=False, format=[0, 1, 2]):
        # mode: train/valid/test
        if path is None:
            return None

        print('Reading {} triples....'.format(mode))
        heads = []
        tails = []
        rels = []
        with open(path) as f:
            if skip_first_line:
                _ = f.readline()
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
        else:
            assert False, "Unknown dataset {}".format(data_name)
    else:
        dataset = KGDatasetToy(data_path, args=args)

    return dataset


class TrainSampler(Dataset):
    """Use Dataset to pack train_sampler
    """

    def __init__(self, edges, n_entities, neg_sample_size):
        super().__init__()
        self.edges = edges
        self.n_entities = n_entities
        self.neg_sample_size = neg_sample_size

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, index):
        h, r, t = self.edges[index]
        return h, r, t

    def head_collate_fn(self, data):
        # Corrupt head
        neg_head = None
        h_new = np.random.randint(0, self.n_entities, self.neg_sample_size)
        unique_entity = np.unique(
            np.concatenate([[x[0] for x in data], [x[2]
                                                   for x in data], h_new]))
        reindex_dict = {}
        for idx in range(len(unique_entity)):
            reindex_dict[unique_entity[idx]] = idx

        def mp(entry):
            return reindex_dict[entry]

        mp = np.vectorize(mp)
        h = mp([x[0] for x in data])
        t = mp([x[2] for x in data])
        h_new = mp(h_new)

        h = paddle.to_tensor(h, dtype='int32')
        r = paddle.to_tensor([x[1] for x in data], dtype='int32')
        t = paddle.to_tensor(t, dtype='int32')
        h_new = paddle.to_tensor(h_new, dtype='int32')
        unique_entity = paddle.to_tensor(unique_entity, dtype='int32')

        return (h, r, t), (h_new, r, t), unique_entity, 1

    def tail_collate_fn(self, data):
        # Corrupt tail
        neg_head = False
        t_new = np.random.randint(0, self.n_entities, self.neg_sample_size)
        unique_entity = np.unique(
            np.concatenate([[x[0] for x in data], [x[2]
                                                   for x in data], t_new]))
        reindex_dict = {}
        for idx in range(len(unique_entity)):
            reindex_dict[unique_entity[idx]] = idx

        def mp(entry):
            return reindex_dict[entry]

        mp = np.vectorize(mp)
        h = mp([x[0] for x in data])
        t = mp([x[2] for x in data])
        t_new = mp(t_new)

        h = paddle.to_tensor(h, dtype='int32')
        r = paddle.to_tensor([x[1] for x in data], dtype='int32')
        t = paddle.to_tensor(t, dtype='int32')
        t_new = paddle.to_tensor(t_new, dtype='int32')
        unique_entity = paddle.to_tensor(unique_entity, dtype='int32')

        return (h, r, t), (h, r, t_new), unique_entity, 0


class TrainDataset(object):
    """Sampler for training.
    """

    def __init__(self, dataset, args, has_importance=False):
        self.edges = dataset.train.T  # numpy.ndarray
        self.n_entities = dataset.n_entities
        num_train = len(self.edges)
        print('|Train|:', num_train)

    def create_sampler(self,
                       batch_size,
                       num_workers,
                       neg_sample_size,
                       neg_mode='head'):
        dataset = TrainSampler(self.edges, self.n_entities, neg_sample_size)
        if neg_mode == 'head':
            sampler = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                collate_fn=dataset.head_collate_fn)
        elif neg_mode == 'tail':
            sampler = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                collate_fn=dataset.tail_collate_fn)
        else:
            assert False, "Not supported neg mode!"
        return sampler


class NewBidirectionalOneShotIterator:
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            pos_triples, neg_triples, ids, neg_head = next(self.iterator_head)
        else:
            pos_triples, neg_triples, ids, neg_head = next(self.iterator_tail)
        return pos_triples, neg_triples, ids, neg_head

    @staticmethod
    def one_shot_iterator(dataloader):
        while True:
            for pos_triples, neg_triples, ids, neg_head in dataloader:
                yield pos_triples, neg_triples, ids, neg_head


class EvalSampler(object):
    """Sampler for validation and testing.
    """

    def __init__(self, edges, batch_size, mode):
        self.edges = edges
        self.batch_size = batch_size
        self.mode = mode
        self.neg_head = 'head' in mode
        self.cnt = 0
        if 'head' in self.mode:
            self.mode = 't,r->h'
            self.num_edges = len(self.edges['t,r->h']['tr'])
        elif 'tail' in self.mode:
            self.mode = 'h,r->t'
            self.num_edges = len(self.edges['h,r->t']['hr'])

    def __iter__(self):
        return self

    def __next__(self):
        if self.cnt == self.num_edges:
            raise StopIteration
        beg = self.cnt
        if self.cnt + self.batch_size > self.num_edges:
            self.cnt = self.num_edges
        else:
            self.cnt += self.batch_size
        if self.mode == 't,r->h':
            return paddle.to_tensor(
                self.edges['t,r->h']['tr'][beg:self.cnt],
                "int64"), paddle.to_tensor(
                    self.edges['t,r->h']['h_correct_index'][beg:self.cnt],
                    "int64"), paddle.to_tensor(
                        self.edges['t,r->h']['h_candidate'][beg:self.cnt],
                        "int64")
        elif self.mode == 'h,r->t':
            return paddle.to_tensor(
                self.edges['h,r->t']['hr'][beg:self.cnt],
                "int64"), paddle.to_tensor(
                    self.edges['h,r->t']['t_correct_index'][beg:self.cnt],
                    "int64"), paddle.to_tensor(
                        self.edges['h,r->t']['t_candidate'][beg:self.cnt],
                        "int64")

    def reset(self):
        """Reset the sampler
        """
        self.cnt = 0
        return self


class EvalDataset(object):
    """Dataset for validation or testing.
    """

    def __init__(self, dataset, args):
        self.name = args.dataset
        src = [dataset.train[0]]
        etype_id = [dataset.train[1]]
        dst = [dataset.train[2]]
        self.num_train = len(dataset.train[0])
        if dataset.valid is not None:
            src.append(dataset.valid[0])
            etype_id.append(dataset.valid[1])
            dst.append(dataset.valid[2])
            self.num_valid = len(dataset.valid[0])
        elif self.name in ['wikikg90m', 'toy']:
            self.valid_dict = dataset.valid_dict
            self.num_valid = len(self.valid_dict['h,r->t']['hr'])
        else:
            self.num_valid = 0
        if dataset.test is not None:
            src.append(dataset.test[0])
            etype_id.append(dataset.test[1])
            dst.append(dataset.test[2])
            self.num_test = len(dataset.test[0])
        elif self.name in ['wikikg90m']:
            self.test_dict = dataset.test_dict
            self.num_test = len(self.test_dict['h,r->t']['hr'])
        else:
            self.num_test = 0
        src = np.concatenate(src)
        etype_id = np.concatenate(etype_id)
        dst = np.concatenate(dst)

    def get_edges(self, eval_type):
        if eval_type == 'valid':
            return self.valid
        elif eval_type == 'test':
            return self.test
        else:
            raise Exception('get invalid type: ' + eval_type)

    def get_dicts(self, eval_type):
        if eval_type == 'valid':
            return self.valid_dict
        elif eval_type == 'test':
            return self.test_dict
        else:
            raise Exception('get invalid type: ' + eval_type)

    def create_sampler(self,
                       eval_type,
                       batch_size,
                       mode='tail',
                       num_workers=32,
                       rank=0,
                       ranks=1):
        """Create sampler for validation or testing
        """
        edges = self.get_dicts(eval_type)
        new_edges = {}
        assert 'tail' in mode
        if 'tail' in mode:
            beg = edges['h,r->t']['hr'].shape[0] * rank // ranks
            end = min(edges['h,r->t']['hr'].shape[0] * (rank + 1) // ranks,
                      edges['h,r->t']['hr'].shape[0])
            new_edges['h,r->t'] = {
                'hr': edges['h,r->t']['hr'][beg:end],
                't_candidate': edges['h,r->t']['t_candidate'][beg:end],
            }
            if 't_correct_index' in edges['h,r->t']:
                new_edges['h,r->t']['t_correct_index'] = edges['h,r->t'][
                    't_correct_index'][beg:end]
            else:
                new_edges['h,r->t']['t_correct_index'] = np.zeros(
                    end - beg, dtype=np.short)
        else:
            assert False, mode
        print(beg, end)
        return EvalSampler(new_edges, batch_size, mode)
