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
import json
from collections import defaultdict

import numpy as np


class TriGraph(object):
    """
    Implementation of knowledge graph interface in pglke.

    Args:
        triplets (dict):
            Triplet data of knowledge graphs with keys 'train', 'valid' and 'test'.
        num_ents (int, optional):
            Number of entities in a knowledge graph.
            If not provided, it will be inferred from triplets.
        num_rels (int, optional):
            Number of relations in a knowledge graph.
            If not provided, it will be inferred from triplets.
        ent_feat (np.ndarray, optional):
            Entity features, which have consistent order with entity ids.
        rel_feat (np.ndarray, optional):
            Relation features, which have consistent order with relation ids.
    """

    def __init__(self,
                 triplets,
                 num_ents=None,
                 num_rels=None,
                 ent_feat=None,
                 rel_feat=None,
                 **kwargs):
        assert isinstance(
            triplets, dict
        ), 'Triplets should be a directionary with keys "train", "valid" and "test".'
        self._train = triplets.get('train', None)
        self._valid = triplets.get('valid', None)
        self._test = triplets.get('test', None)

        if num_ents is None:
            self._num_ents = self.maybe_num_ents()
        else:
            self._num_ents = num_ents

        if num_rels is None:
            self._num_rels = self.maybe_num_rels()
        else:
            self._num_rels = num_rels

        self._ent_feat = ent_feat
        self._rel_feat = rel_feat

    def __repr__(self):
        """Pretty Print the TriGraph.
        """
        repr_dict = {'class': self.__class__.__name__}
        repr_dict['num_ents'] = self._num_ents
        repr_dict['num_rels'] = self._num_rels
        repr_dict['ent_feat'] = []
        repr_dict['rel_feat'] = []
        repr_dict['num_train'] = self._train.shape[0]
        repr_dict['num_valid'] = self._valid['r'].shape[0] if self._valid is not None else 0
        repr_dict['num_test'] = self._test['r'].shape[0] if self._test is not None else 0
        repr_dict['ent_feat'] = self._ent_feat.shape(
        ) if self._ent_feat is not None else -1
        repr_dict['rel_feat'] = self._rel_feat.shape(
        ) if self._rel_feat is not None else -1

        return json.dumps(repr_dict, ensure_ascii=False)

    def maybe_num_ents(self):
        """Count the number of entities.
        """
        ents = []

        def extract_ents(data):
            """Extract entities from data.
            """
            if isinstance(data, dict):
                ents.append(np.unique(data['h']))
                if data['mode'] is not 'wikikg90m':
                    ents.append(np.unique(data['t']))
                else:
                    ents.append(np.unique(data['t_candidate'].reshape((-1, ))))
            elif isinstance(data, np.ndarray):
                ents_data = np.concatenate([data[:, 0], data[:, 2]])
                ents.append(np.unique(ents_data))

        extract_ents(self._train)
        extract_ents(self._valid)
        extract_ents(self._test)
        num_ents = np.unique(np.concatenate(ents)).shape[0]
        return num_ents

    def maybe_num_rels(self):
        """Count the number of relations.
        """
        rels = []

        def extract_rels(data):
            """Extract relations from data.
            """
            if isinstance(data, dict):
                rels.append(np.unique(data['r']))
            elif isinstance(data, np.ndarray):
                rels.append(np.unique(data[:, 1]))

        extract_rels(self._train)
        extract_rels(self._valid)
        extract_rels(self._test)
        num_rels = np.unique(np.concatenate(rels)).shape[0]
        return num_rels

    @classmethod
    def load(cls, path, mmap_mode='r'):
        """
        Load dumped TriGraph from path and return a TriGraph.

        Args:
            path (str):
                Directory of saved TriGraph.
            mmap_mode (str, optional):
                Whether load embeddings and features in mmap_mode.
                Default :code:`mmap_mode="r"`.

        """
        num_ents = np.load(os.path.join(path, 'num_ents.npy'))
        num_rels = np.load(os.path.join(path, 'num_rels.npy'))
        train = np.load(os.path.join(path, 'train.npy'), mmap_mode=mmap_mode)
        valid = np.load(os.path.join(path, 'valid.npy'), mmap_mode=mmap_mode)
        test = np.load(os.path.join(path, 'test.npy'), mmap_mode=mmap_mode)
        ent_feat = np.load(
            os.path.join(path, 'ent_feat.npy'), mmap_mode=mmap_mode)
        rel_feat = np.load(
            os.path.join(path, 'rel_feat.npy'), mmap_mode=mmap_mode)
        triplets = {'train': train, 'valid': valid, 'test': test}
        return cls(triplets, num_ents, num_rels, ent_feat, rel_feat)

    def dump(self, path):
        """
        Dump knowledge graph data into the given directory.

        Args:
            path (str):
                Directory for the storage of the knowledge graph.

        """
        if not os.path.exists(path):
            os.makedirs(path)

        np.save(os.path.join(path, 'num_ents.npy'), self._num_ents)
        np.save(os.path.join(path, 'num_rels.npy'), self._num_rels)
        np.save(os.path.join(path, 'train.npy'), self._train)
        np.save(os.path.join(path, 'valid.npy'), self._valid)
        np.save(os.path.join(path, 'test.npy'), self._test)
        np.save(os.path.join(path, 'ent_feat.npy'), self._ent_feat)
        np.save(os.path.join(path, 'rel_feat.npy'), self._rel_feat)

    @property
    def true_tails_for_head_rel(self):
        """
        Get valid tail entities for a pair of (head, reltion) in KGs.

        Return:
            dict: The dictionary of valid tails for all existing head-relation pairs.
        """
        true_pairs = defaultdict(set)
        for h, r, t in self.triplets:
            true_pairs[(h, r)].add(t)
        for k, v in true_pairs.items():
            true_pairs[k] = np.array(list(v), dtype='int32')
        return true_pairs

    @property
    def true_heads_for_tail_rel(self):
        """
        Get valid head entities for a pair of (tail, relation) in KGs.

        Return:
            dict: The dictionary of valid heads for all existing tail-relation pairs.
        """
        true_pairs = defaultdict(set)
        for h, r, t in self.triplets:
            true_pairs[(t, r)].add(h)
        for k, v in true_pairs.items():
            true_pairs[k] = np.array(list(v), dtype='int32')
        return true_pairs

    @property
    def ent_feat(self):
        """Entity features (np.ndarray).
        """
        return self._ent_feat

    @property
    def rel_feat(self):
        """Relation features (np.ndarray).
        """
        return self._rel_feat

    @property
    def num_triplets(self):
        """Number of all existing triplets.
        """
        return self.num_train + self.num_valid + self.num_test

    @property
    def num_train(self):
        """Number of train triplets.
        """
        if self._train is None:
            return 0
        else:
            return self._train.shape[0]

    @property
    def num_valid(self):
        """Number of valid triplets.
        """
        if self._valid is None:
            return 0
        else:
            if isinstance(self._valid, dict):
                return self._valid['h'].shape[0]
            else:
                return self._valid.shape[0]

    @property
    def num_test(self):
        """Number of test triplets.
        """
        if self._test is None:
            return 0
        else:
            if isinstance(self._test, dict):
                return self._test['h'].shape[0]
            else:
                return self._test.shape[0]

    @property
    def num_ents(self):
        """Number of entities.
        """
        return self._num_ents

    @property
    def num_rels(self):
        """Number of relations.
        """
        return self._num_rels

    @property
    def train_triplets(self):
        """Training triplets (np.ndarray).
        """
        return self._train

    @property
    def valid_dict(self):
        """Validation triplets (dict).
        """
        return self._valid

    @property
    def test_dict(self):
        """Test triplets (dict).
        """
        return self._test

    @property
    def triplets(self):
        """All existing triplets (np.ndarray).
        """
        valid = np.stack(
            [self._valid['h'], self._valid['r'], self._valid['t']],
            axis=1) if self._valid is not None and self._valid[
                'mode'] != 'wikikg90m' else None
        test = np.stack(
            [self._test['h'], self._test['r'], self._test['t']],
            axis=1) if self._test is not None and self._test[
                'mode'] != 'wikikg90m' else None
        unempty = [x for x in [self._train, valid, test] if x is not None]
        return np.concatenate(unempty, axis=0)

    def sorted_triplets(self, sort_by='h'):
        """
        Get sorted training triplets with different strategies.

        Args:
            sorted_by (str):
                The axis as sort key. 'h': axis=0; 'r': axis=1; 't': axis=2.
        Return:
            np.ndarray: The sorted training triplets.

        """
        if sort_by == 'h':
            sort_key = lambda x: (x[0], x[1])
        elif sort_by == 't':
            sort_key = lambda x: (x[2], x[1])
        else:
            sort_key = lambda x: x[1]
        return np.stack(sorted(self._train, key=sort_key))

    @property
    def ents(self):
        """All entity ids from 0 to :code:`num_ents - 1` (np.ndarray).
        """
        return np.arange(0, self._num_ents)

    @property
    def rels(self):
        """All relation ids from 0 to :code:`num_rels - 1` (np.ndarray).
        """
        return np.arange(0, self._num_rels)

    def to_mmap(self, path='./tmp'):
        """Save all data and load embeddings and features in mmap mode.
        """
        self.dump(path)
        new_object = self.load(path)
        self._train = new_object._train
        self._valid = new_object._valid
        self._test = new_object._test
        self._ent_feat = new_object._ent_feat
        self._rel_feat = new_object._rel_feat

    def sampled_subgraph(self, percent, dataset='all'):
        """
        Randomly sample :code:`percent`% of original triplets inplace.

        Args:
            percent (float):
                The percent of triplets sampled.
            dataset (str, optional: 'train', 'valid', 'test', 'all'):
                The triplet data to be sampled.
        """

        def _sample_subgraph(data):
            if isinstance(data, dict):
                for key in data:
                    data[key] = _sample_subgraph(data[key])
            elif isinstance(data, np.ndarray):
                max_index = int(data.shape[0] * percent)
                data = data[:max_index]
            return data

        if dataset == 'train' or dataset == 'all':
            self._train = _sample_subgraph(self._train)
        if dataset == 'valid' or dataset == 'all':
            self._valid = _sample_subgraph(self._valid)
        if dataset == 'test' or dataset == 'all':
            self._test = _sample_subgraph(self._test)

    def random_partition(self, k, mode='train'):
        """
        Randomly divide ids of triplets into k parts.

        Args:
            k (int):
                The number of partitions.
            mode (str):
                The triplet data to be divided.
        Return:
            list of np.ndarray: A list of k groups of triplets' ids.
        """
        assert k > 0 and k <= self.num_train
        p_size = self.num_train // k
        p_more = self.num_train % k
        indexs = np.random.permutation(self.num_train)
        l_part = indexs[:p_more * (p_size + 1)].reshape(p_more, -1)
        r_part = indexs[p_more * (p_size + 1):].reshape(-1, p_size)
        indexs = list(l_part) + list(r_part)

        return indexs
