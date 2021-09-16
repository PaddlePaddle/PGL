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


class KnowlGraph(object):
    """Implementation of knowledge graph interface in pglke

    This is a simple implementation of knowledge graph structure in pglke

    Args:

        triplets: a dictionary of triplets (keys: 'train', 'valid', 'test'). 
                    Each values in the dictionary are (h, r, t) tuples, 2D numpy.array 

        num_ents (optional: int): Number of entities in a knowledge graph.
                    If not provided, it will be inferred from triplets.

        num_rels (optional: int): Number of relations in a knowledge graph.
                    If not provided, it will be inferred from triplets.

        ent_feat (optional): a dictionary of 2D numpy.array as entity features.
                    It should have consistent order with entity ids.

        rel_feat (optional): a dictionary of 2D numpy.array as relation features
                    It should have consistent order with relation ids.

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
        ), 'triplets should be a directionary with keys "train", "valid" and "test"'
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

        def remap_test_data(data):
            """Reformulate validation and test
            """
            if isinstance(data, dict):
                data = {
                    'mode': 'wiki',
                    'h': data['hr'][:, 0],
                    'r': data['hr'][:, 1],
                    'candidate': data['t_candidate'],
                    'correct_index': data['t_correct_index']
                }
            elif isinstance(data, np.ndarray):
                data = {
                    'mode': 'normal',
                    'h': data[:, 0],
                    'r': data[:, 1],
                    't': data[:, 2],
                    'candidate': self._num_ents
                }
            elif data is not None:
                raise ValueError('Unsupported format for validation or test!')
            return data

        self._valid = remap_test_data(self._valid)
        self._test = remap_test_data(self._test)

        if ent_feat is None:
            self._ent_feat = {}
        else:
            self._ent_feat = ent_feat

        if rel_feat is None:
            self._rel_feat = {}
        else:
            self._rel_feat = rel_feat

    def __repr__(self):
        """Pretty Print the KnowledgeGraph
        """
        repr_dict = {'class': self.__class__.__name__}
        repr_dict['num_ents'] = self._num_ents
        repr_dict['num_rels'] = self._num_rels
        repr_dict['ent_feat'] = []
        repr_dict['num_train'] = self._train.shape[0]
        repr_dict['num_valid'] = self._valid['r'].shape[0]
        repr_dict['num_test'] = self._test['r'].shape[0]
        for key, value in self._ent_feat.items():
            repr_dict['ent_feat'].append({
                'name': key,
                'shape': list(value.shape),
                'dtype': str(value.dtype)
            })
        repr_dict['rel_feat'] = []
        for key, value in self._rel_feat.items():
            repr_dict['rel_feat'].append({
                'name': key,
                'shape': list(value.shape),
                'dtype': str(value.dtype)
            })
        return json.dumps(repr_dict, ensure_ascii=False)

    def maybe_num_ents(self):
        """Count the number of entities
        """
        ents = []

        def extract_ents(data):
            """Extract entities from data
            """
            if isinstance(data, dict):
                ents.append(np.unique(data['hr'][:, 0]))
                ents.append(np.unique(data['t_candidate'].reshape(-1)))
            elif isinstance(data, np.ndarray):
                ents_data = np.concatenate([data[:, 0], data[:, 2]])
                ents.append(np.unique(ents_data))

        extract_ents(self._train)
        extract_ents(self._valid)
        extract_ents(self._test)
        num_ents = np.unique(np.concatenate(ents)).shape[0]
        return num_ents

    def maybe_num_rels(self):
        """Count the number of relations
        """
        rels = []

        def extract_rels(data):
            """Extract relations from data
            """
            if isinstance(data, dict):
                rels.append(np.unique(data['hr'][:, 1]))
            elif isinstance(data, np.ndarray):
                rels.append(np.unique(data[:, 1]))

        extract_rels(self._train)
        extract_rels(self._valid)
        extract_rels(self._test)
        num_rels = np.unique(np.concatenate(rels)).shape[0]
        return num_rels

    @classmethod
    def load(cls, path, mmap_mode='r'):
        """Load KnowledgeGraph from path and return a KnowldegeGraph.

        Args:

            path: The dictionary path of the stored KnowledgeGraph.

            mmap_mode: Default :code:`mmap_mode="r"`. If not None, memory-map the knowledge graph.

        """
        num_ents = np.load(os.path.join(path, 'num_ents.npy'))
        num_rels = np.load(os.path.join(path, 'num_rels.npy'))
        train = np.load(os.path.join(path, 'train.npy'), mmap_mode=mmap_mode)
        valid = np.load(os.path.join(path, 'valid.npy'), mmap_mode=mmap_mode)
        test = np.load(os.path.join(path, 'test.npy'), mmap_mode=mmap_mode)

        def _load_feat(feat_path):
            feat = {}
            if os.path.isdir(feat_path):
                for feat_name in os.listdir(feat_path):
                    feat[os.path.splitext(feat_name)[0]] = np.load(
                        os.path.join(feat_path, feat_name),
                        mmap_mode=mmap_mode)
            return feat

        ent_feat = _load_feat(os.path.join(path, 'ent_feat'))
        rel_feat = _load_feat(os.path.join(path, 'rel_feat'))
        triplets = {'train': train, 'valid': valid, 'test': test}
        return cls(triplets, num_ents, num_rels, ent_feat, rel_feat)

    def dump(self, path):
        """Dump the knowledge graph into a directory.

        This function will dump the knowledge graph information into the given directory path.
        The graph can be read back with :code:`pglke.KnowledgeGraph.load`

        Args:
            path: The directory for the storage of the knowledge graph.

        """
        if not os.path.exists(path):
            os.makedirs(path)

        np.save(os.path.join(path, 'num_ents.npy'), self._num_ents)
        np.save(os.path.join(path, 'num_rels.npy'), self._num_rels)
        np.save(os.path.join(path, 'train.npy'), self._train)
        np.save(os.path.join(path, 'valid.npy'), self._valid)
        np.save(os.path.join(path, 'test.npy'), self._test)

        def _dump_feat(feat_path, feat):
            if len(feat) == 0:
                return

            if not os.path.exists(feat_path):
                os.mkdirs(feat_path)
            for key, value in feat.items():
                np.save(os.path.join(feat_path, key + '.npy'), value)

        _dump_feat(path, 'ent_feat')
        _dump_feat(path, 'rel_feat')

    @property
    def pos_t4hr(self):
        """Return a dict of numpy array, key:(h, r), value:{t}
        """
        true_pairs = defaultdict(set)
        for h, r, t in self.triplets:
            true_pairs[(h, r)].add(t)
        for k, v in true_pairs.items():
            true_pairs[k] = np.array(list(v))
        return true_pairs

    @property
    def pos_h4tr(self):
        """Return a dict of numpy array, key:(t, r), value:{h}
        """
        true_pairs = defaultdict(set)
        for h, r, t in self.triplets:
            true_pairs[(t, r)].add(h)
        for k, v in true_pairs.items():
            true_pairs[k] = np.array(list(v))
        return true_pairs

    @property
    def ent_feat(self):
        """Return numpy.array of entity features.
        """
        return self._ent_feat

    @property
    def rel_feat(self):
        """Return numpy.array of relation features.
        """
        return self._rel_feat

    @property
    def num_triplets(self):
        """Return the number of triplets.
        """
        return self.num_train + self.num_valid + self.num_test

    @property
    def num_train(self):
        """Return the number of train triplets.
        """
        if self._train is None:
            return 0
        else:
            return self._train.shape[0]

    @property
    def num_valid(self):
        """Return the number of valid triplets.
        """
        if self._valid is None:
            return 0
        else:
            if isinstance(self._valid, dict):
                return self._valid.values()[0].shape[0]
            else:
                return self._valid.shape[0]

    @property
    def num_test(self):
        """Return the number of test triplets.
        """
        if self._test is None:
            return 0
        else:
            if isinstance(self._test, dict):
                return self._test.values()[0].shape[0]
            else:
                return self._test.shape[0]

    @property
    def num_ents(self):
        """Return the number of entities.
        """
        return self._num_ents

    @property
    def num_rels(self):
        """Return the number of relations.
        """
        return self._num_rels

    @property
    def train(self):
        """Return train triplets in numpy.ndarray with shape (num_triplets, 3).
        """
        return self._train

    @property
    def valid(self):
        """Return valid triplets in dictionary.
            {'mode': (optional: 'tail', 'both'),
             'h': numpy.ndarray with shape(num_valid,), 
             'r': numpy.ndarray with shape(num_valid,),
             't': numpy.ndarray with shape(num_valid,),  
             'condidate': int or numpy.ndarray with shape (num_valid, 1001),
             'correct_index': numpy.ndarray with shape(num_valid,),}.
        """
        return self._valid

    @property
    def test(self):
        """Return test triplets in dictionary.
            It has the same keys as valid.
        """
        return self._test

    @property
    def triplets(self):
        """Return all triplets in numpy.ndarray with shape (num_triplets, 3).
        """
        unempty = [
            x for x in [self._train, self._valid, self._test] if x is not None
        ]
        return np.concatenate(unempty)

    def sorted_triplets(self, sort_by='h', train_only=True):
        """Return sorted triplets with different strategies.

        This function will return sorted triplets with different strategies.
        If :code:`sort_by='h', then edges will be sorted by head, axis=0 in :code:`triplets`;
        If :code:`sort_by='r', then edges will be sorted by relation, axis=1 in :code:`triplets`;
        If :code:`sort_by='t', then edges will be sorted by tail, axis=2 in :code:`triplets`.

        Args:

            sorted_by: The type for sorted triplets. (optional: 'h', 'r', 't')

        Return:

            A list of (sorted_h, sorted_r, sorted_t), numpy.ndarray

        """
        if sort_by == 'h':
            sort_key = lambda x: (x[0], x[1])
        elif sort_by == 't':
            sort_key = lambda x: (x[2], x[1])
        else:
            sort_key = lambda x: x[1]

        if train_only:
            return np.stack(sorted(self._train, key=sort_key))

    @property
    def ents(self):
        """Return all entities id from 0 to :code:`num_ents - 1`.
        """
        return np.arange(0, self._num_ents)

    @property
    def rels(self):
        """Return all relations id from 0 to :code:`num_rels - 1`.
        """
        return np.arange(0, self._num_rels)

    def to_mmap(self, path='./tmp'):
        """Turn the Knowledge Graph into Memmap mode which can share memory between processes.
        """
        self.dump(path)
        new_object = self.load(path)
        self._train = new_object._train
        self._valid = new_object._valid
        self._test = new_object._test
        self._ent_feat = new_object._ent_feat
        self._rel_feat = new_object._rel_feat

    def sampled_subgraph(self, percent, dataset='all'):
        """Randomly sample :code:`percent`% of original triplets inplace.

        Args:

            percent: The percent of triplets sampled.

            dataset (optional: 'train', 'valid', 'test', 'all'): The key of triplets to be sampled.

        """

        def _sample_subgraph(data):
            if isinstance(data, dict):
                for key in data:
                    data[key] = _sample_subgraph(data[key])
            elif isinstance(data, np.ndarray):
                max_index = data.shape[0]
                index = np.random.permutation(max_index)
                index = index[:int(max_index * percent)]
                data = data[index]
            return data

        if dataset == 'train' or dataset == 'all':
            self._train = _sample_subgraph(self._train)
        if dataset == 'valid' or dataset == 'all':
            self._valid = _sample_subgraph(self._valid)
        if dataset == 'test' or dataset == 'all':
            self._test = _sample_subgraph(self._test)

    def random_partition(self, k, mode='train'):
        """Return a list of k randomly partitioned triplets, each iterm is numpy.ndarray for training

        Args:

            k: The number of partitions.

        """
        assert k > 0 and k <= self.num_train
        p_size = self.num_train // k
        p_more = self.num_train % k
        indexs = np.random.permutation(self.num_train)
        l_part = indexs[:p_more * (p_size + 1)].reshape(p_more, -1)
        r_part = indexs[p_more * (p_size + 1):].reshape(-1, p_size)
        indexs = list(l_part) + list(r_part)

        return indexs
