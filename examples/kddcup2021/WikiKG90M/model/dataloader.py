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
import pdb
import math
import pickle

import numpy as np
import paddle
from paddle.io import Dataset, DataLoader


def uniform_sampler(k, candidate, filter_set=None):
    """
    args k: nagative sample size
    """
    if filter_set is not None:
        new_e_list = []
        new_e_num = 0
        while new_e_num < k:
            new_e = np.random.randint(0, candidate, 2 * k)
            mask = np.in1d(new_e, filter_set, invert=True)
            new_e = new_e[mask]
            new_e_list.append(new_e)
            new_e_num += len(new_e)
        new_e = np.concatenate(new_e_list)[:k]
    else:
        new_e = np.random.randint(0, candidate, k)
    return new_e


def group_index(data):
    """
    Function to reindex elements in data
    args data: a list of elements
    return:
        reindex_dict - a reindex function to apply to a list
        uniques - unique elements in data
    """
    uniques = np.unique(np.concatenate(data))
    reindex_dict = dict([(x, i) for i, x in enumerate(uniques)])
    reindex_func = np.vectorize(lambda x: reindex_dict[x])
    return reindex_func, uniques


class TrainSampler(Dataset):
    """Use Dataset to pack train_sampler
    """

    def __init__(self,
                 edges,
                 n_entities,
                 neg_sample_size,
                 fhead=None,
                 ftail=None):
        super().__init__()
        self.edges = edges
        self.n_entities = n_entities
        self.neg_sample_size = neg_sample_size
        self.fhead = fhead
        self.ftail = ftail

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, index):
        h, r, t = self.edges[index]
        return h, r, t

    def head_collate_fn(self, data):
        # Corrupt head
        neg_head = True
        h, r, t = np.array(data).T

        neg_h = uniform_sampler(self.neg_sample_size, self.n_entities,
                                self.fhead)
        reindexer, uniques = group_index([h, t, neg_h])

        h = paddle.to_tensor(reindexer(h), dtype='int64')
        r = paddle.to_tensor(r, dtype='int64')
        t = paddle.to_tensor(reindexer(t), dtype='int64')
        neg_h = paddle.to_tensor(reindexer(neg_h), dtype='int64')
        uniques = paddle.to_tensor(uniques, dtype='int64')

        return (h, r, t), neg_h, uniques, 1

    def tail_collate_fn(self, data):
        # Corrupt tail
        neg_head = False
        h, r, t = np.array(data).T

        neg_t = uniform_sampler(self.neg_sample_size, self.n_entities,
                                self.ftail)
        reindexer, uniques = group_index([h, t, neg_t])

        h = paddle.to_tensor(reindexer(h), dtype='int64')
        r = paddle.to_tensor(r, dtype='int64')
        t = paddle.to_tensor(reindexer(t), dtype='int64')
        neg_t = paddle.to_tensor(reindexer(neg_t), dtype='int64')
        uniques = paddle.to_tensor(uniques, dtype='int64')

        return (h, r, t), neg_t, uniques, 0


class NewBidirectionalOneShotIterator:
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            pos_triples, neg_ents, ids, neg_head = next(self.iterator_head)
        else:
            pos_triples, neg_ents, ids, neg_head = next(self.iterator_tail)
        return pos_triples, neg_ents, ids, neg_head

    @staticmethod
    def one_shot_iterator(dataloader):
        while True:
            for pos_triples, neg_ents, ids, neg_head in dataloader:
                yield pos_triples, neg_ents, ids, neg_head


class TrainDataset(object):
    """Sampler for training.
    """

    def __init__(self, dataset, args, has_importance=False):
        self.edges = dataset.train
        self.n_entities = dataset.n_entities
        self.fhead = dataset.head_pair
        self.ftail = dataset.tail_pair
        num_train = len(self.edges)
        print('|Train|:', num_train)

    def random_partition(self, k):
        '''
        k: number of partitions
        '''
        num_train = len(self.edges)
        assert k > 0 and k <= num_train

        part_size = num_train // k
        indexs = np.random.permutation(num_train)
        indexs = [
            indexs[i * part_size:min((i + 1) * part_size, num_train)]
            for i in range(k)
        ]
        return indexs

    def create_sampler(self,
                       batch_size,
                       num_workers,
                       neg_sample_size,
                       neg_mode='head',
                       edge_index=None):
        if edge_index is None:
            edges = self.edges
        else:
            edges = self.edges[edge_index]
        dataset = TrainSampler(edges, self.n_entities, neg_sample_size,
                               self.fhead, self.ftail)
        if neg_mode == 'head':
            sampler = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=num_workers,
                collate_fn=dataset.head_collate_fn)
        elif neg_mode == 'tail':
            sampler = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=num_workers,
                collate_fn=dataset.tail_collate_fn)
        elif neg_mode == 'bi-oneshot':
            head_sampler = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=num_workers,
                collate_fn=dataset.head_collate_fn)
            tail_sampler = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=num_workers,
                collate_fn=dataset.tail_collate_fn)
            sampler = NewBidirectionalOneShotIterator(head_sampler,
                                                      tail_sampler)
        else:
            assert False, "Not supported neg mode!"
        return sampler


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


class EvalDataset1(object):
    """Dataset for validation or test on full E set
    """

    def __init__(self, dataset, args):
        pass


class EvalDataset(object):
    """Dataset for validation or testing in wiki format.
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
        elif self.name in [
                'wikikg90m', 'toy', 'FB15k-237', 'FB15k', 'wikikg2'
        ]:
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


if __name__ == '__main__':
    print('no filter')
    print(uniform_sampler(3, 10))

    print('with filter')
    fset = np.array([0, 1, 2])
    for i in range(10):
        print(uniform_sampler(3, 10, True, fset))
