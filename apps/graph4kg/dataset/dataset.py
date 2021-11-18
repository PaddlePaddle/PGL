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

        num_ents: number of entities in a knowledge graph, int

        args: arguments of sampling, argparse.Namespace, including
            - neg_sample_size: Number of negative samples for each triplet, int

            - neg_sample_type: The strategy used for negative sampling.
                'batch': sampling from entities in batch;
                'full': sampling entities from the whole entity set.
                'chunk': sampling from the whole entity set as chunks 

            - filter_sample: Whether filter out valid triplets in knowledge graphs

        filter_dict: a dictionary of valid triplets, in the form of
                    {'head': {(t, r):set(h)}, 'tail': {(h, r):set(t)}}

        shared_path: a dictionary of numpy embeddings' path for embedding prefetch, 
                in the form of {'ent': ent_path, 'rel': rel_path}. None means no prefetch.

    """

    def __init__(self,
                 triplets,
                 num_ents,
                 args,
                 filter_dict=None,
                 shared_path=None):
        shared_ent_path = shared_path.get(
            'ent', None) if shared_path is not None else None
        shared_rel_path = shared_path.get(
            'rel', None) if shared_path is not None else None

        self._triplets = triplets
        self._num_ents = num_ents
        self._neg_sample_size = args.neg_sample_size
        self._neg_sample_type = args.neg_sample_type
        self._sample_weight = args.sample_weight
        if self._sample_weight is True:
            assert filter_dict is not None
            assert 'head' in filter_dict
            assert 'tail' in filter_dict
            self._filter_dict = filter_dict
        else:
            self._filter_dict = {'head': None, 'tail': None}

        self._filter_sample = args.filter_sample
        if self._filter_sample is True:
            raise NotImplementedError('sampling with positive triplets '\
                'filtered is not implemented!')

        self._step = 0

        self._ent_embedding = None
        self._rel_embedding = None
        if shared_ent_path is not None:
            self._ent_embedding = np.load(shared_ent_path, mmap_mode='r+')
        if shared_rel_path is not None:
            self._rel_embedding = np.load(shared_rel_path, mmap_mode='r+')

    def __len__(self):
        return len(self._triplets)

    def __getitem__(self, index):
        h, r, t = self._triplets[index]
        if self._sample_weight:
            weight = self.create_sample_weight(h, r, t)
        else:
            weight = -1
        return h, r, t, weight

    def collate_fn(self, data):
        """Collate_fn to corrupt heads and tails by turns
        """
        self._step = self._step ^ 1
        if self._step == 0:
            return self._collate_fn(data, 'head', self._filter_dict['head'])
        else:
            return self._collate_fn(data, 'tail', self._filter_dict['tail'])

    def _collate_fn(self, data, mode, fl_set):
        h, r, t, weights = np.array(data).T

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

        if self._ent_embedding is not None:
            all_ents_emb = self._ent_embedding[all_ents].astype(np.float32)
        else:
            all_ents_emb = None

        if self._rel_embedding is not None:
            r_emb = self._rel_embedding[r].astype(np.float32)
        else:
            r_emb = None

        h = reindex_func(h)
        t = reindex_func(t)
        neg_ents = reindex_func(neg_ents)

        if weights.sum() < 0:
            weights = None

        indexs = (h, r, t, neg_ents, all_ents)
        embeds = (all_ents_emb, r_emb, weights)
        return indexs, embeds, mode

    def create_sample_weight(self, head, rel, tail):
        """Create weights for samples like RotatE
        """
        weights = max(len(self._filter_dict['head'][(head, rel)]) + \
            len(self._filter_dict['tail'][(tail, rel)]), 1.)
        weights = np.sqrt(1. / weights)
        return weights

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

        triplets: a list of (h, r, t) tuples, 2D numpy.ndarray

        num_ents: number of entities in a knowledge graph, int

    """

    def __init__(self, triplets, num_ents):
        self._num_ents = num_ents
        self._h = triplets['h']
        self._r = triplets['r']
        self._t = triplets['t']

    def __len__(self):
        return self._h.shape[0]

    def __getitem__(self, index):
        h = self._h[index]
        r = self._r[index]
        t = self._t[index]
        return h, r, t


class TestWikiKG2(Dataset):
    """Test Dataset for OGBL-WikiKG2

    Args:

        triplets: a dictionary of triplets with keys
            'h', 'r', 't', 'candaidate_h' and 'candidate_t'
    """

    def __init__(self, triplets):
        self._h = triplets['h']
        self._r = triplets['r']
        self._t = triplets['t']
        self._neg_head = triplets['candidate_h']
        self._neg_tail = triplets['candidate_t']

    def __len__(self):
        return self._h.shape[0]

    def __getitem__(self, index):
        h = self._h[index]
        r = self._r[index]
        t = self._t[index]
        neg_head = self._neg_head[index]
        neg_tail = self._neg_tail[index]
        return h, r, t, neg_head, neg_tail


class TestWikiKG90M(Dataset):
    """Test Dataset for WikiKG90M

    Args:

        triplets: a dictionary of triplets with keys 'h', 'r', 'candidate_t', 't_correct_index'
    """

    def __init__(self, triplets):
        self._h = triplets['h']
        self._r = triplets['r']
        self._candidate = triplets['candidate_t']
        self._t_index = triplets.get('t_correct_index', None)

    def __len__(self):
        return self._h.shape[0]

    def __getitem__(self, index):
        h = self._h[index]
        r = self._r[index]
        neg_tail = self._candidate[index]
        if self._t_index is not None:
            t = self._t_index[index]
            return h, r, t, neg_tail
        else:
            return h, r, -1, neg_tail


def create_dataloaders(trigraph, args, filter_dict=None, shared_ent_path=None):
    """Construct DataLoader for training, validation and test
    """
    train_dataset = KGDataset(
        triplets=trigraph.train_triplets,
        num_ents=trigraph.num_ents,
        args=args,
        filter_dict=filter_dict if args.filter_sample else None,
        shared_path={'ent': shared_ent_path} if args.mix_cpu_gpu else None)

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
        if args.data_name == 'wikikg90m':
            valid_dataset = TestWikiKG90M(trigraph.valid_dict)
        elif args.data_name == 'wikikg2':
            valid_dataset = TestWikiKG2(trigraph.valid_dict)
        else:
            valid_dataset = TestKGDataset(trigraph.valid_dict,
                                          trigraph.num_ents)
        valid_loader = DataLoader(
            dataset=valid_dataset, batch_size=args.test_batch_size)
    else:
        valid_loader = None

    if args.test:
        if args.data_name == 'wikikg90m':
            test_dataset = TestWikiKG90M(trigraph.test_dict)
        elif args.data_name == 'wikikg2':
            test_dataset = TestWikiKG2(trigraph.test_dict)
        else:
            test_dataset = TestKGDataset(trigraph.test_dict, trigraph.num_ents)
        test_loader = DataLoader(
            dataset=test_dataset, batch_size=args.test_batch_size)
    else:
        test_loader = None

    return train_loader, valid_loader, test_loader
