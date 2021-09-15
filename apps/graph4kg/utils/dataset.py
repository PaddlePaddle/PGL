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

import numpy as np
import paddle
from paddle.io import Dataset, DataLoader

from helper import timer_wrapper


class KGDataset(Dataset):
    """Implementation of Dataset for knowledge graphs

    Args:

        triplets:  a list of (h, r, t) tuples, 2D numpy.ndarray or
                a dictionary of triplets as validation and test set of WikiKG90M-LSC

        num_ents: Number of entities in a knowledge graph, int

        num_negs: Number of negative samples for each triplet, int

        neg_mode: The strategy used for negative sampling.
                'batch': sampling from entities in batch;
                'full': sampling entities from the whole entity set.

        filter_mode: Whether filter out valid triplets in knowledge graphs

        filter_dict: a dictionary of valid triplets, in the form of
                    {'head': {(t, r):set(h)}, 'tail': {(h, r):set(t)}}

    """
    def __init__(self,
                 triplets,
                 num_ents,
                 num_negs,
                 neg_mode='batch',
                 filter_mode=False,
                 filter_dict=None):
        self._triplets = triplets
        self._num_ents = num_ents
        self._num_negs = num_negs
        self._neg_mode = neg_mode
        self._filter_mode = filter_mode
        if self._filter_mode == False:
            self._filter_dict = {'head': None, 'tail': None}
        else:
            self._filter_dict = filter_dict
            assert self._filter_dict is not None
            assert 'head' in self._filter_dict
            assert 'tail' in self._filter_dict

    def __len__(self):
        return len(self._triplets)

    def __getitem__(self, index):
        return self._triplets[index]

    def head_collate_fn(self, data):
        """Collate_fn to corrupt heads
        """
        return self._collate_fn(data, 'head', self._filter_dict['head'])

    def tail_collate_fn(self, data):
        """Collate_fn to corrupt tails
        """
        return self._collate_fn(data, 'tail', self._filter_dict['tail'])

    def _collate_fn(self, data, mode, fl_set):
        h, r, t = np.array(data).T
        if self._neg_mode == 'batch':
            reindex_func, all_ents = self.group_index([h, t])
            cand = all_ents
        elif self._neg_mode == 'full':
            cand = self._num_ents
        else:
            raise ValueError('neg_mode % not supported!' % self._neg_mode)

        neg_ents = []
        if mode == 'head':
            for hi, ri, ti in data:
                neg_ents.append(self.uniform_sampler(
                    self._num_negs, cand, fl_set[(ti, ri)] 
                ))
        else:
            for hi, ri, ti in data:
                neg_ents.append(self.uniform_sampler(
                    self._num_negs, cand, fl_set[(hi, ri)] 
                ))

        if self._neg_mode == 'full':
            ents_list = [h, t, np.concatenate(neg_ents)]
            reindex_func, all_ents = self.group_index(ents_list)

        h = paddle.to_tensor(reindex_func(h), dtype='int64')
        r = paddle.to_tensor(r, dtype='int64')
        t = paddle.to_tensor(reindex_func(t), dtype='int64')
        neg_ents = np.stack([reindex_func(x) for x in neg_ents])
        neg_ents = paddle.to_tensor(neg_ents, dtype='int64')
        neg_ents = paddle.to_tensor(neg_ents, dtype='int64')
        all_ents = paddle.to_tensor(all_ents, dtype='int64')

        return (h, r, t, neg_ents), all_ents, mode

    @staticmethod
    def group_index(data):
        """Function to reindex elements in data.
        Args data: a list of elements
        Return:
            reindex_dict - a reindex function to apply to a list
            uniques - unique elements in data
        """
        uniques = np.unique(np.concatenate(data))
        reindex_dict = dict([(x, i) for i, x in enumerate(uniques)])
        reindex_func = np.vectorize(lambda x: reindex_dict[x])
        return reindex_func, uniques

    @staticmethod
    def uniform_sampler(k, cand, filter_set=None):
        """Sampling negative samples uniformly.
        Args k: nagative sample size
        """
        n_cand = cand if isinstance(cand, int) else len(cand)
        e_cand = None if isinstance(cand, int) else cand
        if filter_set is not None:
            new_e_list = []
            new_e_num = 0
            while new_e_num < k:
                new_e = np.random.randint(0, n_cand, 2 * k)
                new_e = new_e if e_cand is None else e_cand[new_e]
                mask = np.in1d(new_e, filter_set, invert=True)
                new_e = new_e[mask]
                new_e_list.append(new_e)
                new_e_num += len(new_e)
            new_e = np.concatenate(new_e_list)[:k]
        else:
            new_e = np.random.randint(0, n_cand, k)
            new_e = new_e if e_cand is None else e_cand[new_e]
        return new_e


class TestKGDataset(Dataset):
    """Implementation of Dataset for triplets in dict format

    Args:

        triplets: a dictionary of triplets as validation and test set of WikiKG90M-LSC or
                a list of (h, r, t) tuples, 2D numpy.ndarray

        num_ents: Number of entities in a knowledge graph, int

    """
    def __init__(self,
                 triplets,
                 num_ents):
        self._num_ents = num_ents
        self._mode = self.triplets['mode']
        assert self._mode in ['tail', 'both']
        self._h = triplets['h']
        self._r = triplets['r']
        self._t = triplets.get('t', None)
        self._cand = triplets['candidate']
        self._corr_idx = triplets.get('correct_index', None)

    def __len__(self):
        return len(self.triplets['r'])

    def __getitem__(self, index):
        sample = {}
        sample['h'] = self._h[index]
        sample['r'] = self._r[index]
        sample['t'] = self._t[index] if self._t else None
        if self.mode == 'both':
            sample['cand'] = self._cand
        else:
            sample['cand'] = self._cand[index]
            if self._corr_idx:
                sample['corr_idx'] = self._corr_idx[index]
            else:
                sample['corr_idx'] = None
        return sample

    def collate_fn(self, data):
        """Collate_fn for validation and test set
        """
        mode = self._mode
        h = paddle.to_tensor([x['h'] for x in data])
        r = paddle.to_tensor([x['r'] for x in data])
        if mode == 'both':
            t = paddle.to_tensor([x['t'] for x in data])
            cand = paddle.reshape(paddle.arange(data[0]['cand']), (1, -1))
            corr_idx = None
        else:
            t = None
            cand = paddle.to_tensor(np.stack([x['cand'] for x in data]))
            if self._corr_idx:
                corr_idx = paddle.to_tensor(
                    np.stack([x['corr_idx'] for x in data]))
            else:
                corr_idx = None
        return mode, (h, r, t, cand), corr_idx


class KGDataLoader(object):
    """Implementation of DataLoader for knowledge graphs

    Args:

        dataset: a KGDataset object storing triplets

        batch_size: Batch size for loading samples

        num_workers: Number of processes for loading samples

        sample_mode (optional: 'head', 'tail', 'bi-oneshot'): The way to corrupt triplets

    """ 

    def __init__(self,
                 dataset,
                 batch_size,
                 shuffle=True,
                 num_workers=4,
                 drop_last=True,
                 sample_mode='tail'):
        def _create_loader(collate_fn):
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                num_workers=num_workers,
                collate_fn=collate_fn
            )
        self.mode = sample_mode
        if self.mode == 'bi-oneshot':
            self.loader = [_create_loader(dataset.head_collate_fn),
                           _create_loader(dataset.tail_collate_fn)]
        else:
            if self.mode == 'head':
                collate_fn = dataset.head_collate_fn
            elif self.mode == 'tail':
                collate_fn = dataset.tail_collate_fn
            elif self.mode == 'test':
                collate_fn = dataset.collate_fn
            else:
                raise ValueError('%d sample mode not supported!' % self.mode)
            self.loader = _create_loader(collate_fn)
            
        self.step = 0

    def __next__(self):
        if self.mode == 'head' or self.mode == 'tail':
            return next(self.loader)
        else:
            self.step += 1
            return next(self.loader[self.step % 2])

    def __iter__(self):
        return self
