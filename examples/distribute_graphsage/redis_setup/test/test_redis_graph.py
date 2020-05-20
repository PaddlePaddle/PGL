#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
# File: test_redis_graph.py
# Author: suweiyue(suweiyue@baidu.com)
# Date: 2019/08/19 16:28:18
#
########################################################################
"""
    Comment.
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import sys

import numpy as np
import tqdm
from pgl.redis_graph import RedisGraph

if __name__ == '__main__':
    host, port = sys.argv[1].split(':')
    port = int(port)
    redis_configs = [{"host": host, "port": port}, ]
    graph = RedisGraph("reddit-graph", redis_configs, num_parts=64)
    #nodes = np.arange(0, 100)
    #for i in range(0, 100):
    for l in tqdm.tqdm(sys.stdin):
        l_sp = l.rstrip().split('\t')
        if len(l_sp) != 2:
            continue
        i, j = int(l_sp[0]), int(l_sp[1])
        nodes = graph.sample_predecessor(np.array([i]), 10000)
        assert j in nodes

