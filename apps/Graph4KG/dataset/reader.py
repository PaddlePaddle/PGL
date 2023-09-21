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
import time

import numpy as np
import paddle.distributed as dist

from dataset.trigraph import TriGraph
from utils import timer_wrapper


class TripletDataset(object):
    """
    Load knowledge graph data from files

    Args:
        path (str):
            Directory of triplet dataset.
        data_name (str):
            The folder name of triplet dataset.
        hrt_mode (str, optional):
            The order of head, relation and tail per line in triplet file.
            Choices: 'hrt', 'htr', 'trh', 'thr', 'rht', 'rth'.
        kv_mode (str, optional):
            The order of string names and ids in dictionary files.
            'kv' denotes entity_name/relation_name, id.
        train_file (str, optional):
            Filename of training data.
        valid_file (str, optional):
            Filename of validation data.
        test_file (str, optional):
            Filename of test data.
        ent_file (str, optional):
            Filename of entity_to_id dictionary.
        rel_file (str, optional):
            Filename of relation_to_id dictionary.
        delimiter (char, optional):
            The delimiter in files.
        map_to_id (bool, optional):
            Whether to map loaded elements into int ids.
        load_dict (bool, optional):
            Whether to load dictionaries from files.
        skip_head (bool, optional):
            Whether to ignore the first line of dictionary files.
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
                 load_dict=False,
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
        self._load_dict = load_dict

        if not os.path.exists(self._path):
            raise ValueError('data path %s not exists!' % self._path)

        ent_dict_path = os.path.join(self._path, ent_file)
        rel_dict_path = os.path.join(self._path, rel_file)
        if not self._load_dict:
            self.create_dict(ent_dict_path, rel_dict_path)
        self._ent_dict = self.load_dictionary(
            ent_dict_path, self._kv, self._delimiter, self._skip_head)
        self._rel_dict = self.load_dictionary(
            rel_dict_path, self._kv, self._delimiter, self._skip_head)
        self.num_ents = len(self._ent_dict)
        self.num_rels = len(self._rel_dict)

        self.ent_feat = None
        self.rel_feat = None
        self.train, self.valid, self.test = self.load_dataset()
        self.triplets = {
            'train': self.train,
            'valid': {
                'mode': 'hrt',
                'h': self.valid[:, 0],
                'r': self.valid[:, 1],
                't': self.valid[:, 2],
                'candidate': self.num_ents
            },
            'test': {
                'mode': 'hrt',
                'h': self.test[:, 0],
                'r': self.test[:, 1],
                't': self.test[:, 2],
                'candidate': self.num_ents
            }
        }

    def __call__(self):
        return self.graph

    @staticmethod
    def load_dictionary(path, kv_mode, delimiter='\t', skip_head=False):
        """Load dictionary from file, an item per line.
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
        """Load triplets from file, a triplet per line.
        """
        if not os.path.exists(path):
            raise ValueError('there is no triplet file in %s' % path)
        map_fn = lambda x: [x[hrt_idx[i]] for i in range(3)]
        with open(path, 'r') as rp:
            rp.readline() if skip_head else None
            data = [map_fn(l.strip().split(delimiter)) for l in rp.readlines()]
        return data

    def create_dict(self, ent_path, rel_path):
        while not (os.path.exists(ent_path) and os.path.exists(rel_path)):
            if dist.get_rank() == 0:
                data = []
                for file in self._data_list:
                    path = os.path.join(self._path, file)
                    data.append(
                        self.load_triplet(path, self._hrt, self._delimiter,
                                          self._skip_head))
                all_ents = set()
                all_rels = set()
                for triplets in data:
                    for h, r, t in triplets:
                        all_ents.add(h)
                        all_ents.add(t)
                        all_rels.add(r)
                self._ent_dict = dict([(x, i) for i, x in enumerate(all_ents)])
                self._rel_dict = dict([(x, i) for i, x in enumerate(all_rels)])
                if not os.path.exists(ent_path):
                    with open(ent_path, 'w') as fp:
                        for k, v in self._ent_dict.items():
                            fp.write('{}\t{}\n'.format(k, v))
                if not os.path.exists(rel_path):
                    with open(rel_path, 'w') as fp:
                        for k, v in self._rel_dict.items():
                            fp.write('{}\t{}\n'.format(k, v))
            else:
                time.sleep(1)

    def load_dataset(self):
        """Load datasets from files, including train, value, test.
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
                """Cast elements in x into int type.
                """
                return [int(i) for i in x]

        for i, sub_data in enumerate(data):
            sub_data = [map_fn(x) for x in sub_data]
            data[i] = np.array(sub_data, dtype=np.int64)

        return data


class WikiKG90MDataset(object):
    """Load WikiKG90M from files.
    """

    def __init__(self, path):
        super(WikiKG90MDataset, self).__init__()
        self.name = 'WikiKG90M-LSC'
        try:
            from ogb.lsc import WikiKG90MDataset as LSCDataset
        except ImportError as error:
            print(
                'Please run ``pip install ogb==1.3.1`` to load WikiKG90M dataset.'
            )
            raise ImportError(error)
        data = LSCDataset(path)
        valid = data.valid_dict['h,r->t']
        test = data.test_dict['h,r->t']
        self.triplets = {
            'train': data.train_hrt,
            'valid': {
                'mode': 'wikikg90m',
                'h': valid['hr'][:, 0],
                'r': valid['hr'][:, 1],
                'candidate_t': valid['t_candidate'],
                't_correct_index': valid.get('t_correct_index', None)
            },
            'test': {
                'mode': 'wikikg90m',
                'h': test['hr'][:, 0],
                'r': test['hr'][:, 1],
                'candidate_t': test['t_candidate'],
                't_correct_index': test.get('t_correct_index', None)
            }
        }
        self.num_ents = data.num_entities
        self.num_rels = data.num_relations
        self.ent_feat = data.entity_feat
        self.rel_feat = data.relation_feat


class WikiKG2Dataset(object):
    """Load OGBL-WikiKG2 from files.
    """

    def __init__(self, path):
        super(WikiKG2Dataset, self).__init__()
        self.name = 'OGBL-WikiKG2'
        try:
            from ogb.linkproppred import LinkPropPredDataset
        except ImportError as error:
            print(
                'Please run ``pip install ogb==1.3.1`` to load OGBL-WikiKG2 dataset.'
            )
            raise ImportError(error)
        data = LinkPropPredDataset(name='ogbl-wikikg2', root=path)
        split_idx = data.get_edge_split()
        valid = split_idx['valid']
        test = split_idx['test']
        self.triplets = {
            'train': np.stack([
                split_idx['train']['head'], split_idx['train']['relation'],
                split_idx['train']['tail']
            ]).T,
            'valid': {
                'mode': 'wikikg2',
                'h': valid['head'],
                'r': valid['relation'],
                't': valid['tail'],
                'candidate_h': valid['head_neg'],
                'candidate_t': valid['tail_neg']
            },
            'test': {
                'mode': 'wikikg2',
                'h': test['head'],
                'r': test['relation'],
                't': test['tail'],
                'candidate_h': test['head_neg'],
                'candidate_t': test['tail_neg']
            }
        }
        self.num_ents = data.graph['num_nodes']
        self.num_rels = int(max(data.graph['edge_reltype'])[0]) + 1
        self.ent_feat = None
        self.rel_feat = None


def read_trigraph(data_path, data_name, use_dict, kv_mode):
    """Load datasets from files.
    """
    if data_name == "wikikg90m":
        dataset = WikiKG90MDataset(data_path)
    elif data_name == 'wikikg2':
        dataset = WikiKG2Dataset(data_path)
    elif data_name in ['FB15k-237', 'WN18RR', 'FB15k', 'wn18']:
        dataset = TripletDataset(
            data_path,
            data_name,
            map_to_id=True,
            load_dict=use_dict,
            kv_mode=kv_mode)
    else:
        raise NotImplementedError('Please add %s to read_trigraph function '
                                  'in dataset/reader.py to load this dataset' %
                                  data_name)

    graph_data = TriGraph(dataset.triplets, dataset.num_ents, dataset.num_rels,
                          dataset.ent_feat, dataset.rel_feat)

    return graph_data
