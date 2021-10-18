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
from paddle.io import Dataset
from numpy.random import default_rng

from utils.helper import timer_wrapper
from models.embedding import NumPyEmbedding


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
                 filter_dict=None,
                 shared_ent_path=None,
                 shared_rel_path=None):
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
        self._step = 0
        self._ent_embedding = self._load_mmap_embedding(shared_ent_path)
        self._rel_embedding = self._load_mmap_embedding(shared_rel_path)

    def __len__(self):
        return len(self._triplets)

    def __getitem__(self, index):
        h, r, t = self._triplets[index]
        return h, r, t

    def __getitem__1(self, index):
        h, r, t = self._triplets[index]
        if self._filter_mode:
            if self._step == 0:
                filter_set = self._filter_dict['head'][(t, r)]
            else:
                filter_set = self._filter_dict['tail'][(h, r)]
            negs = self.uniform_sampler(self._num_negs, self._num_ents,
                                        filter_set)
        else:
            negs = np.random.randint(0, self._num_ents, self._num_negs)
        negs = np.reshape(negs, -1)
        return h, r, t, negs

    def _load_mmap_embedding(self, path):
        if path is not None:
            return NumPyEmbedding(weight_path=path, load_mode=True)
        return None

    def collate_fn(self, data):
        """Collate_fn to corrupt heads and tails by turns
        """
        self._step = self._step ^ 1
        if self._step == 0:
            return self._collate_fn(data, 'head', self._filter_dict['head'])
        else:
            return self._collate_fn(data, 'tail', self._filter_dict['tail'])

    def collate_fn_1(self, data):
        """Collate_fn to corrupt heads and tails by turns through __getitem__
        """
        h = np.array([x[0] for x in data])
        r = np.array([x[1] for x in data])
        t = np.array([x[2] for x in data])
        negs = np.concatenate([x[3] for x in data])
        reindex_func, all_ents = self.group_index([h, t, negs])
        if self._ent_embedding is not None:
            all_ents_emb = self._ent_embedding.get(all_ents).astype(np.float32)
        else:
            all_ents_emb = None

        if self._rel_embedding is not None:
            r_emb = self._rel_embedding.get(r).astype(np.float32)
        else:
            r_emb = None
        h = reindex_func(h)
        t = reindex_func(t)
        neg_ents = reindex_func(negs)
        neg_ents = neg_ents.reshape((-1, self._num_negs))
        mode = 'head' if self._step == 0 else 'tail'
        self._step = self._step ^ 1

        return (h, r, t, neg_ents, all_ents), (r_emb, all_ents_emb), mode

    def _collate_fn_(self, data, mode, fl_set):
        h = np.array([6984, 9882, 9378, 444, 3384])
        t = np.array([3862, 6778, 6537, 7253, 5209])
        neg_ents = np.array([13449, 10019, 2784, 5991, 9498])
        r = np.array([118, 401, 236, 1236, 765])
        index = np.argsort(np.concatenate([h, t, neg_ents]))
        reindex_func, all_ents = self.group_index([h, t, neg_ents])
        h = reindex_func(h)
        t = reindex_func(t)

        neg_ents = paddle.to_tensor(reindex_func(neg_ents), dtype='int64')
        neg_ents = neg_ents.reshape((-1, self._num_negs))
        all_ents_emb = np.concatenate([
            np.load(
                '/ssd2/wanghuijuan03/githubs/dgl-ke/python/dglke/p_ent_emb.npy'
            ), np.load(
                '/ssd2/wanghuijuan03/githubs/dgl-ke/python/dglke/neg_tail.npy')
        ])[index]
        r_emb = self._rel_embedding.get(r).astype(np.float32)
        r_emb = np.load(
            '/ssd2/wanghuijuan03/githubs/dgl-ke/python/dglke/rel_emb.npy')
        return (h, r, t, neg_ents, all_ents), (r_emb, all_ents_emb), mode

    def _collate_fn(self, data, mode, fl_set):
        h, r, t = np.array(data).T
        if self._neg_mode == 'batch':
            reindex_func, all_ents = self.group_index([h, t])
            cand = all_ents
        elif self._neg_mode == 'full':
            cand = self._num_ents
        else:
            raise ValueError('neg_mode %s not supported!' % self._neg_mode)

        if fl_set is None:
            cand = len(cand) if self._neg_mode == 'batch' else cand
            neg_ents = self.uniform_sampler(self._num_negs * len(data), cand)
            if self._neg_mode == 'full':
                reindex_func, all_ents = self.group_index([h, t, neg_ents])
                neg_ents = reindex_func(neg_ents)
            neg_ents = neg_ents.reshape(-1, self._num_negs)
        else:
            neg_ents = []
            if mode == 'head':
                for hi, ri, ti in data:
                    fl_set_i = fl_set[(ti, ri)] if fl_set else None
                    neg_ents.append(
                        self.uniform_sampler(self._num_negs, cand, fl_set_i))
            else:
                for hi, ri, ti in data:
                    fl_set_i = fl_set[(hi, ri)] if fl_set else None
                    neg_ents.append(
                        self.uniform_sampler(self._num_negs, cand, fl_set_i))

            if self._neg_mode == 'full':
                ents_list = [h, t, np.concatenate(neg_ents)]
                reindex_func, all_ents = self.group_index(ents_list)
            neg_ents = np.stack([reindex_func(x) for x in neg_ents])

        if self._ent_embedding is not None:
            all_ents_emb = self._ent_embedding.get(all_ents).astype(np.float32)
        else:
            all_ents_emb = None

        if self._rel_embedding is not None:
            r_emb = self._rel_embedding.get(r).astype(np.float32)
        else:
            r_emb = None

        h = reindex_func(h)
        t = reindex_func(t)

        return (h, r, t, neg_ents, all_ents), (r_emb, all_ents_emb), mode

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
        rng = default_rng(0)
        n_cand = cand if isinstance(cand, int) else len(cand)
        e_cand = None if isinstance(cand, int) else cand
        if filter_set is not None:
            new_e_list = []
            new_e_num = 0
            while new_e_num < k:
                new_e = rng.choice(n_cand, 2 * k, replace=False)
                new_e = new_e if e_cand is None else e_cand[new_e]
                mask = np.in1d(new_e, filter_set, invert=True)
                new_e = new_e[mask]
                new_e_list.append(new_e)
                new_e_num += len(new_e)
            new_e = np.concatenate(new_e_list)[:k]
        else:
            new_e = rng.choice(n_cand, k, replace=False)
            new_e = new_e if e_cand is None else e_cand[new_e]
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
        h = paddle.to_tensor([x[0][0] for x in data])
        r = paddle.to_tensor([x[0][1] for x in data])
        if mode == 'wiki':
            t = None
            cand = paddle.to_tensor(np.stack([x[1] for x in data]))
            if self._corr_idx:
                corr_idx = np.stack([x[2] for x in data])
            else:
                corr_idx = None
        elif mode == 'normal':
            t = paddle.to_tensor([x[0][2] for x in data])
            cand = paddle.reshape(paddle.arange(data[0][1]), (1, -1))
            corr_idx = None
        return mode, (h, r, t, cand), corr_idx
