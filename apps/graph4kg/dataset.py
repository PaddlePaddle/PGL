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

import pdb
import numpy as np

from knowlgraph import KnowlGraph
from utils.helper import timer_wrapper

__all__ = ['TripletDataset', 'WikiKG90MDataset']


class TripletDataset(object):
    """ Load a knowledge graph in triplets

    Attributes:
        ent_dict: store data values - {entity:id}
        rel_dict: store data values - {relation:id}
        n_ents: number of entities - int
        n_rels: number of relations - int
        train: a triplet per sample - [[h, r, t]]
        valid: query, candidate, answer per sample - ([[h,r]],[[e]],[[t]])
        test: query, candidate, answer per sample - ([[h,r]],[[e]],[[t]])
        head_pair: set of h appear with (t, r) pair - {(t,r):set(h)}
        tail_pair: set of t appear with (h, r) pair - {(h,r):set(t)}
        entity_feat: n-dim features for entities - [[x1, x2, ..., xn]]
        relation_feat: n-dim features for relations - [[x1, x2, ..., xn]]
    """

    def __init__(self,
                 path,
                 data_name,
                 hrt_mode='hrt',
                 kv_mode='kv',
                 train_file='train.txt',
                 valid_file='valid.txt',
                 test_file='test.txt',
                 ent_file='entities.dict',
                 rel_file='relations.dict',
                 ent_feat_path=None,
                 rel_feat_path=None,
                 delimiter='\t',
                 map_to_id=False,
                 skip_head=False):
        super(TripletDataset, self).__init__()
        self._path = os.path.join(path, data_name)
        hrt_dict = dict([x, i] for i, x in enumerate(hrt_mode))
        self._hrt = [hrt_dict[x] for x in ['h', 'r', 't']]
        self._kv = True if kv_mode == 'kv' else False
        self._name = data_name
        self._data_list = [train_file, valid_file, test_file]
        self._feat_path = [ent_feat_path, rel_feat_path]
        self._delimiter = delimiter
        self._map_to_id = map_to_id
        self._skip_head = skip_head

        if not os.path.exists(self._path):
            raise ValueError('data path %s not exists!' % self._path)

        # TODO(Huijuan): No dictionary
        if self._map_to_id:
            ent_dict_path = os.path.join(self._path, ent_file)
            rel_dict_path = os.path.join(self._path, rel_file)
            self._ent_dict = self.load_dictionary(
                ent_dict_path, self._kv, self._delimiter, self._skip_head)
            self._rel_dict = self.load_dictionary(
                rel_dict_path, self._kv, self._delimiter, self._skip_head)
            self.num_ents = len(self._ent_dict)
            self.num_rels = len(self._rel_dict)
        else:
            self.num_ents = None
            self.num_rels = None

        self.ent_feat = None
        self.rel_feat = None
        self.train, self.valid, self.test = self.load_dataset()
        self.graph = self.create_knowlgraph()

    def __call__(self):
        return self.graph

    @staticmethod
    def load_dictionary(path, kv_mode, delimiter='\t', skip_head=False):
        """Function to load dictionary from file, an item per line.
        """
        if not os.path.exists(path):
            raise ValueError('there is no dictionary file in %s' % path)
        step = 1 if kv_mode else -1
        map_fn = lambda x: [x[::step][0], int(x[::step][1])]
        with open(path, 'r') as rp:
            rp.readline() if skip_head else None
            data = [map_fn(l.strip().split(delimiter)) for l in rp.readlines()]
        return dict(data)

    @staticmethod
    def load_triplet(path, hrt_idx, delimiter='\t', skip_head=False):
        """Function to load triplets from file, a triplet per line.
        """
        if not os.path.exists(path):
            raise ValueError('there is no triplet file in %s' % path)
        map_fn = lambda x: [x[hrt_idx[i]] for i in range(3)]
        with open(path, 'r') as rp:
            rp.readline() if skip_head else None
            data = [map_fn(l.strip().split(delimiter)) for l in rp.readlines()]
        return data

    def load_dataset(self):
        """Function to load datasets from files, including train, value, test
        """
        data = []
        for file in self._data_list:
            path = os.path.join(self._path, file)
            data.append(
                self.load_triplet(path, self._hrt, self._delimiter,
                                  self._skip_head))

        if self._map_to_id:

            def map_fn(x):
                """Map entities and relations into ids.
                """
                h = self._ent_dict[x[0]]
                r = self._rel_dict[x[1]]
                t = self._ent_dict[x[2]]
                return [h, r, t]
        else:

            def map_fn(x):
                """Case ids of entities and relations into int.
                """
                return [int(i) for i in x]

        for i, sub_data in enumerate(data):
            sub_data = [map_fn(x) for x in sub_data]
            data[i] = np.array(sub_data, dtype=np.int64)

        return data

    def create_knowlgraph(self):
        """Return the knowledge graph as a KnowlGraph object
        """
        triplets = {
            'train': self.train,
            'valid': self.valid,
            'test': self.test
        }
        graph = KnowlGraph(triplets, self.num_ents, self.num_rels,
                           self.ent_feat, self.rel_feat)
        # graph.sampled_subgraph(percent=0.01)
        return graph


class WikiKG90MDataset(object):
    """WikiKG90M dataset implementation
    """

    def __init__(self, path):
        super(WikiKG90MDataset, self).__init__()
        self.name = 'WikiKG90M-LSC'
        from ogb.lsc import WikiKG90MDataset as LSCDataset
        data = LSCDataset(path)
        triplets = {
            'train': data.train_hrt,
            'valid': data.valid_dict['h,r->t'],
            'test': data.test_dict['h,r->t']
        }
        num_ents = data.num_entities
        num_rels = data.num_relations
        ent_feat = {'text': data.entity_feat}
        rel_feat = {'text': data.relation_feat}
        self.graph = KnowlGraph(triplets, num_ents, num_rels, ent_feat,
                                rel_feat)


class WikiKG2Dataset(object):
    """OGBL-WikiKG2 dataset implementation
    """

    def __init__(self, path):
        super(WikiKG2Dataset, self).__init__()
        self.name = 'OGBL-WikiKG2'
        from ogb.linkproppred import LinkPropPredDataset
        data = LinkPropPredDataset(name='ogbl-wikikg2', root=path)
        split_idx = data.get_edge_split()
        triplets = {
            'train': np.stack([
                split_idx['train']['head'], split_idx['train']['relation'],
                split_idx['train']['tail']
            ]).T,
            'valid': np.stack([
                split_idx['valid']['head'], split_idx['valid']['relation'],
                split_idx['valid']['tail']
            ]).T,
            'test': np.stack([
                split_idx['test']['head'], split_idx['test']['relation'],
                split_idx['test']['tail']
            ]).T
        }
        num_ents = data.graph['num_nodes']
        num_rels = int(max(data.graph['edge_reltype'])[0]) + 1
        self.graph = KnowlGraph(triplets, num_ents, num_rels)


@timer_wrapper('dataset loading')
def load_dataset(data_path, data_name):
    """Load datasets from files
    """
    if data_name == "wikikg90m":
        dataset = WikiKG90MDataset(data_path)
    elif data_name == 'wikikg2':
        dataset = WikiKG2Dataset(data_path)
    elif data_name in ['FB15k', 'WN18', 'FB15k-237', 'WN18RR']:
        dataset = TripletDataset(
            data_path, data_name, kv_mode='vk', map_to_id=True)

    return dataset
