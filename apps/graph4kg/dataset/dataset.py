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

import paddle
import numpy as np
from numpy.random import default_rng
from paddle.io import Dataset
from paddle.io import DataLoader, DistributedBatchSampler

from utils import timer_wrapper
from models.numpy_embedding import NumPyEmbedding


class KGDataset(Dataset):
    """Implementation of Dataset for knowledge graphs

    Args:

        triplets:  a list of (h, r, t) tuples, 2D numpy.ndarray or
                a dictionary of triplets as validation and test set of WikiKG90M-LSC

        num_ents: Number of entities in a knowledge graph, int

        neg_sample_size: Number of negative samples for each triplet, int

        neg_sample_type: The strategy used for negative sampling.
                'batch': sampling from entities in batch;
                'full': sampling entities from the whole entity set.
                'chunk': sampling from the whole entity set as chunks 

        num_chunks: Number of chunks of negative samples, int

        filter_sample: Whether filter out valid triplets in knowledge graphs

        filter_dict: a dictionary of valid triplets, in the form of
                    {'head': {(t, r):set(h)}, 'tail': {(h, r):set(t)}}

    """

    def __init__(self,
                 triplets,
                 num_ents,
                 neg_sample_size,
                 neg_sample_type='batch',
                 filter_sample=False,
                 filter_dict=None,
                 shared_ent_path=None,
                 shared_rel_path=None):
        self._prefetch_ent = shared_ent_path is not None
        self._prefetch_rel = shared_rel_path is not None

        self._triplets = triplets
        self._num_ents = num_ents
        self._neg_sample_size = neg_sample_size
        self._neg_sample_type = neg_sample_type
        self._filter_sample = filter_sample
        if self._filter_sample == False:
            self._filter_dict = {'head': None, 'tail': None}
        else:
            raise NotImplementedError('sampling with positive triplets '\
                'filtered is not implemented!')
            self._filter_dict = filter_dict
            assert self._filter_dict is not None
            assert 'head' in self._filter_dict
            assert 'tail' in self._filter_dict
        self._step = 0

        if self._prefetch_ent:
            self._ent_embedding = np.load(shared_ent_path, mmap_mode='r+')
        if self._prefetch_rel:
            self._rel_embedding = np.load(shared_rel_path, mmap_mode='r+')

    def __len__(self):
        return len(self._triplets)

    def __getitem__(self, index):
        h, r, t = self._triplets[index]
        return h, r, t

    def collate_fn(self, data):
        """Collate_fn to corrupt heads and tails by turns
        """
        self._step = self._step ^ 1
        if self._step == 0:
            return self._collate_fn(data, 'head', self._filter_dict['head'])
        else:
            return self._collate_fn(data, 'tail', self._filter_dict['tail'])

    def _collate_fn(self, data, mode, fl_set):
        h, r, t = np.array(data).T
        if self._neg_sample_type == 'batch':
            neg_size = self._neg_sample_size * h.shape[0]
            reindex_func, all_ents = self.group_index([h, t])
            neg_ents = self.uniform_sampler(neg_size, all_ents)

        elif self._neg_sample_type == 'full':
            neg_size = self._neg_sample_size * h.shape[0]
            neg_ents = self.uniform_sampler(neg_size, self._num_ents)
            reindex_func, all_ents = self.group_index([h, t, neg_ents])

        elif self._neg_sample_type == 'chunk':
            neg_size = max(h.shape[0], self._neg_sample_size)
            neg_ents = self.uniform_sampler(neg_size, self._num_ents)
            reindex_func, all_ents = self.group_index([h, t, neg_ents])

        else:
            raise ValueError('neg_sample_type %s not supported!' %
                             self._neg_sample_type)

        if self._prefetch_ent:
            all_ents_emb = self._ent_embedding[all_ents].astype(np.float32)
        else:
            all_ents_emb = None

        if self._prefetch_rel:
            r_emb = self._rel_embedding[r].astype(np.float32)
        else:
            r_emb = None

        h = reindex_func(h)
        t = reindex_func(t)
        neg_ents = reindex_func(neg_ents)

        return (h, r, t, neg_ents, all_ents), (all_ents_emb, r_emb), mode

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
        rng = default_rng()
        if filter_set is not None:
            new_e_list = []
            new_e_num = 0
            while new_e_num < k:
                new_e = rng.choice(cand, 2 * k, replace=True)
                mask = np.in1d(new_e, filter_set, invert=True)
                new_e = new_e[mask]
                new_e_list.append(new_e)
                new_e_num += len(new_e)
            new_e = np.concatenate(new_e_list)[:k]
        else:
            new_e = rng.choice(cand, k, replace=True)
        return new_e


class TestKGDataset(Dataset):
    """Implementation of Dataset for triplets in dict format

    Args:

        triplets: a dictionary of triplets as validation and test set of WikiKG90M-LSC or
                a list of (h, r, t) tuples, 2D numpy.ndarray

        num_ents: Number of entities in a knowledge graph, int

    """

    def __init__(self, triplets, num_ents):
        self._num_ents = num_ents
        self._mode = triplets['mode']
        assert self._mode in ['wiki', 'normal']
        self._h = triplets['h']
        self._r = triplets['r']
        self._t = triplets.get('t', None)
        self._cand = triplets['candidate']
        self._corr_idx = triplets.get('correct_index', None)

    def __len__(self):
        return len(self._r)

    def __getitem__(self, index):
        h = self._h[index]
        r = self._r[index]
        t = self._t[index] if self._t is not None else None
        if self._mode == 'normal':
            cand = self._cand
        else:
            cand = self._cand[index]
        if self._corr_idx:
            corr_idx = self._corr_idx[index]
        else:
            corr_idx = None
        return (h, r, t), cand, corr_idx

    def collate_fn(self, data):
        """Collate_fn for validation and test set
        """
        mode = self._mode
        h = np.array([x[0][0] for x in data])
        r = np.array([x[0][1] for x in data])
        if mode == 'wiki':
            t = None
            cand = np.stack([x[1] for x in data])
            if self._corr_idx:
                corr_idx = np.stack([x[2] for x in data])
            else:
                corr_idx = None
        elif mode == 'normal':
            t = np.array([x[0][2] for x in data])
            cand = np.arange(data[0][1]).reshape((1, -1))
            corr_idx = None
        return mode, (h, r, t, cand), corr_idx


def create_dataloaders(trigraph, args, filter_dict=None, shared_ent_path=None):

    train_dataset = KGDataset(
        triplets=trigraph.train_triplets,
        num_ents=trigraph.num_ents,
        neg_sample_size=args.neg_sample_size,
        neg_sample_type=args.neg_sample_type,
        filter_sample=args.filter_sample,
        filter_dict=filter_dict if args.filter_sample else None,
        shared_ent_path=shared_ent_path if args.mix_cpu_gpu else None)

    train_sampler = DistributedBatchSampler(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=train_dataset.collate_fn)

    if args.valid:
        assert trigraph.valid_dict is not None, 'validation set is not given!'
        valid_dataset = TestKGDataset(
            triplets=trigraph.valid_dict, num_ents=trigraph.num_ents)
        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=args.test_batch_size,
            collate_fn=valid_dataset.collate_fn)
    else:
        valid_loader = None

    if args.test:
        assert trigraph.test_dict is not None, 'test set is not given!'
        test_dataset = TestKGDataset(
            triplets=trigraph.test_dict, num_ents=trigraph.num_ents)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=args.test_batch_size,
            collate_fn=test_dataset.collate_fn)
    else:
        test_loader = None

    return train_loader, valid_loader, test_loader
