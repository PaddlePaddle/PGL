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
from ogb.lsc import WikiKG90MDataset

from utils import timer_wrapper


class KGDataset(object):
    """ Load a knowledge graph

    Attributes:
        * ent_dict: store data values - {entity:id}
        * rel_dict: store data values - {relation:id}
        * n_entities: number of entities - int
        * n_relations: number of relations - int
        * train: a triplet per sample - [[h, r, t]]
        * valid: query, candidate, answer per sample - ([[h,r]],[[e]],[[t]])
        * test: query, candidate, answer per sample - ([[h,r]],[[e]],[[t]])
        * head_pair: set of h appear with (t, r) pair - {(t,r):set(h)}
        * tail_pair: set of t appear with (h, r) pair - {(h,r):set(t)}
        * entity_feat: n-dim features for entities - [[x1, x2, ..., xn]]
        * relation_feat: n-dim features for relations - [[x1, x2, ..., xn]]
    """

    def __init__(self,
                 path,
                 data_name,
                 mode='hrt',
                 data_list=['train.txt', 'valid.txt', 'test.txt'],
                 dict_list=['entities.dict', 'relations.dict'],
                 map_to_id=True):
        super(KGDataset, self).__init__()
        self.path = os.path.join(path, data_name)
        self.mode = mode
        self.name = data_name
        self.data_list = data_list
        self.dict_list = dict_list
        self.map_to_id = map_to_id

        self.get_datasets = {
            'hrt': self.triplet_dataset,
            'wikikg90m': self.wikikg90m_dataset
        }[self.mode]
        self.get_datasets()

        self.head_pair = None
        self.tail_pair = None

    @staticmethod
    def get_subset(data, percent):
        num_sample = int(len(data) * percent)
        if isinstance(data, np.array):
            return data[:num_sample]
        if isinstance(data, dict):
            for k, v in data.items():
                data[k] = v[:num_sample]
            return data

    @staticmethod
    def load_dictionary(path, delimiter='\t', kv_mode=False, skip_head=False):
        assert path is not None, 'dictionary data path is None!'
        step = 1 if kv_mode else -1
        map_fn = lambda x: [x[::step][0], int(x[::step][1])]
        with open(path, 'r') as rp:
            rp.readline() if skip_head else None
            data = [map_fn(l.strip().split(delimiter)) for l in rp.readlines()]
        return dict(data)

    @staticmethod
    def load_triplet(path, delimiter='\t', hrt_idx=[0, 1, 2], skip_head=False):
        assert path is not None, 'triplet data path is None!'
        map_fn = lambda x: [x[hrt_idx[i]] for i in range(3)]
        with open(path, 'r') as rp:
            rp.readline() if skip_head else None
            data = [map_fn(l.strip().split(delimiter)) for l in rp.readlines()]
        return data

    def get_dicts(self):
        data = []
        for file in self.dict_list:
            path = os.path.join(self.path, file)
            data.append(self.load_dictionary(path))
        self.ent_dict, self.rel_dict = data

    def construct_true_pair(self):
        head_pair = defaultdict(set)
        tail_pair = defaultdict(set)

        def add_pair(data):
            for h, r, t in zip(*data):
                head_pair[(t, r)].add(h)
                tail_pair[(h, r)].add(t)

        add_pair(self.train)
        if self.valid is not None:
            add_pair(self.valid)
        if self.test is not None:
            add_pair(self.test)
        self.head_pair = head_pair
        self.tail_pair = tail_pair

    def set_train_subset(self, percent):
        self.train = self.get_subset(self.train, percent)

    def set_valid_subset(self, percent):
        self.valid = self.get_subset(self.valid, percent)

    def set_test_subset(self, percent):
        self.test = self.get_subset(self.test, percent)

    @timer_wrapper('triplets loading')
    def triplet_dataset(self):
        self.get_dicts()
        self.n_entities = len(self.ent_dict)
        self.n_relations = len(self.rel_dict)

        data = []
        for file in self.data_list:
            path = os.path.join(self.path, file)
            data.append(self.load_triplet(path))

        if self.map_to_id:

            def map_fn(x):
                h = self.ent_dict[x[0]]
                r = self.rel_dict[x[1]]
                t = self.ent_dict[x[2]]
                return [h, r, t]

            for i, sub_data in enumerate(data):
                sub_data = [map_fn(x) for x in sub_data]
                data[i] = np.array(sub_data, dtype=np.int64)

        self.entity_feat = None
        self.relation_feat = None

        # for i in range(1,3):
        #     data[i] = {
        #         'hr': data[i][:, :2],
        #         't_index': data[i][:, 2],
        #         'tr': np.stack([data[i][:,2], data[i][:,1]]).T
        #         'h_index': data[i][:, 0]
        #         'candidate': np.arange(self.n_entities)
        #     }

        self.train, self.valid, self.test = data

    @timer_wrapper('wikikg90m loading')
    def wikikg90m_dataset(self):
        data = WikiKG90MDataset(path)
        self.n_entities = data.num_entities
        self.n_relations = data.num_relations
        self.train = data.train_hrt.T
        self.valid = {
            'hr': data.valid_dict['h,r->t']['hr'],
            'candidate': data.valid_dict['h,r->t']['t_candidate'],
            't_index': data.valid_dict['h,r->t']['t_correct_index']
        }
        self.test = {
            'hr': data.test_dict['h,r->t']['hr'],
            'candidate': data.test_dict['h,r->t']['t_candidate']
        }
        self.entity_feat = data.entity_feat
        self.relation_feat = data.relation_feat


def get_dataset(args,
                data_path,
                data_name,
                format_str='built_in',
                delimiter='\t',
                files=None,
                has_edge_importance=False):
    if data_name == "wikikg90m":
        dataset = KGDataset(data_path, data_name, 'wikikg90m')
    else:
        dataset = KGDataset(data_path, data_name, 'hrt')

    return dataset
