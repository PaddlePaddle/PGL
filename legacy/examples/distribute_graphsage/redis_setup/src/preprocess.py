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
import sys

import numpy as np
import scipy.sparse as sp


def _load_config(fn):
    ret = {}
    with open(fn) as f:
        for l in f:
            if l.strip() == '' or l.startswith('#'):
                continue
            k, v = l.strip().split('=')
            ret[k] = v
    return ret


def _prepro(config):
    data = np.load("../data/reddit.npz")
    adj = sp.load_npz("../data/reddit_adj.npz")
    adj = adj.tocoo()
    src = adj.row
    dst = adj.col

    with open(config['edge_path'], 'w') as f:
        for idx, e in enumerate(zip(src, dst)):
            s, d = e
            l = "{}\t{}\t{}\n".format(s, d, idx)
            f.write(l)
    feats = data['feats'].astype(np.float32)
    np.savez(config['node_feat_path'], feats=feats)


if __name__ == '__main__':
    config = _load_config('./redis_graph.cfg')
    _prepro(config)
