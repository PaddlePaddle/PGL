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
from numpy.random import default_rng
import paddle
from paddle.io import Dataset
from paddle.io import DataLoader, DistributedBatchSampler

from utils import timer_wrapper


class KGDataset(Dataset):
    """
    Dataset for knowledge graphs

    Args:
        triplets (list of tuples or 2D numpy.ndarray):
            The collection of training triplets (h, r, t) with shape [num_triplets, 3].
        num_ents (int):
            The number of entities in the knowledge graph.
        args (argparse.Namespace):
            Arguments of negative sampling, including:
            - neg_sample_size (int): Number of negative samples for each triplet.
            - neg_sample_type (str): The strategy used for negative sampling.
                'batch': sampling from current batch.
                'full': sampling from all entities.
                'chunk': triplets are divided into X chunks and each chunk shares
                    a group of negative samples sampled from all entities.
            - filter_sample (bool): Whether filter out existing triplets.
        filter_dict (dict, optional):
            Dictionary of existing triplets, in the form of
            {'head': {(t, r):set(h)}, 'tail': {(h, r):set(t)}}.
            Default to None.
        shared_path (dict, optional):
            Dictionary of shared embeddings' path for embedding prefetch
            in the form of {'ent': ent_path, 'rel': rel_path}.
            Default to None.

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
        self._sample_weight = args.weighted_loss
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
        """Collate_fn to corrupt heads and tails by turns.
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
        """Create weights for samples.
        """
        assert self._filter_dict is not None, 'Can not '\
            'create weights of samples as filter dictionary is not given!'
        weights = max(len(self._filter_dict['head'][(head, rel)]) + \
            len(self._filter_dict['tail'][(tail, rel)]), 1.)
        weights = np.sqrt(1. / weights)
        return weights

    @staticmethod
    def group_index(data):
        """
        Function to reindex elements in data.
        Args:
            data (list): A list of int values.
        Return:
            function: The reindex function to apply to a list.
            np.ndarray: Unique elements in data.
        """
        uniques = np.unique(np.concatenate(data))
        reindex_dict = dict([(x, i) for i, x in enumerate(uniques)])
        reindex_func = np.vectorize(lambda x: reindex_dict[x])
        return reindex_func, uniques

    @staticmethod
    def uniform_sampler(k, cand, filter_set=None):
        """
        Sampling negative samples uniformly.

        Args:
            k (int): Number of sampled elements.
            cand (list or int): The list of elements to sample. The int
                value X denotes sampling integers from [0, X).
            filter_set (list): The list of invalid int values.
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
    """
    Dataset for test triplets in dict format.

    Args:
        triplets (dict):
            The collection of triplets with keys 'h', 'r' and 't'.
            The values are 1D np.ndarray.
        num_ents (int):
            Number of entities.

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
    """
    Dataset for test data in OGBL-WikiKG2

    Args:
        triplets (dict):
            The collections of test data with keys 'h', 'r',
            't', 'candaidate_h' and 'candidate_t'.
            The values of 'h', 'r' and 't' are 1D np.ndarray.
            The values of 'candidate_*' are 2d np.ndarray.
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
    """
    Dataset for test data in WikiKG90M

    Args:
        triplets (dict):
            The collection of test data with keys 'h', 'r',
            'candidate_t', 't_correct_index'.
            The values of 'h', 'r' and 't_correct_index' are 1D np.ndarray.
            The values of 'candidate_t' are 2D np.ndarray.
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
    """Construct DataLoader for training, validation and test.
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
