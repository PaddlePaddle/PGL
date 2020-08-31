from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import sys
import numpy as np
import sentencepiece as spm
import jieba
import re
from propeller import log
from functools import partial
from random import choice
import itertools
import six

rev_dict = {i: j.strip() for i, j in enumerate(open('./data/xnli/teacher_vocab.txt'))}

def debug(i):
    i = '|'.join([rev_dict[ii] for ii in i])
    return i


def build_truncate_augment(min_length_rate=0.8):
    def aug(inputs):
        start = 0
        end = len(inputs)
        l = np.random.randint(int(min_length_rate * end), end - start)
        begin = np.random.randint(0, end - l)
        ret = inputs[begin: begin + l]
        return ret
    return aug


def build_unk_augment(unk_id=1, replace_rate=0.15):
    def aug(inputs):
        l = len(inputs)
        ret = (np.random.rand(l) < replace_rate).astype(np.int64) * inputs
        return ret
    return aug


def build_pos_augment(pos_dict, replace_rate=0.15):
    dict_hash = [set(v) for v in pos_dict.values()]
    dict_tuple = [tuple(v) for v in pos_dict.values()]
    def lookfor_pos_replace(word):
        for s, t in zip(dict_hash, dict_tuple):
            if word in s:
                return choice(t)
        return word

    def aug(inputs):
        log.debug('before pos: %s' % debug(inputs))
        l = len(inputs)
        rate = (np.random.rand(l) < replace_rate).astype(np.int64)
        log.debug(rate)
        ret = [lookfor_pos_replace(i) if r else i for r, i in zip(rate, inputs)]
        log.debug('after pos: %s' % debug(ret))
        return np.array(ret)
    return aug


augment_methods = {
    'unk': build_unk_augment,
    'truncate': build_truncate_augment,
}


def build_random_augment(method='all', ratial=None, args=None):
    if method == 'all':
        method = augment_methods.keys()
    if ratial is None:
        ratial = [1. / len(augment_methods)] * len(augment_methods)
    if args is None:
        args = [{}] * len(augment_methods)

    assert len(method) == len(ratial) == len(args)
    funcs = [augment_methods[m](**arg) for m, arg in zip(method, args)]
    ratial = list(itertools.accumulate(ratial))
    def random_arg(inputs):
        eps = np.random.rand()
        for r, func in zip(ratial, funcs):
            if eps < r:
                ret = func(inputs)
                return ret
    return random_arg

