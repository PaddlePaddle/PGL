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

