#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
# File: encrypt.py
# Author: suweiyue(suweiyue@baidu.com)
# Date: 2019/10/08 15:52:16
#
########################################################################
"""
    Comment.
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import struct
import logging
import argparse
import numpy as np
import collections
from distutils import dir_util
import pickle

#from utils import print_arguments 
import paddle.fluid as F
from paddle.fluid.proto import framework_pb2


log = logging.getLogger(__name__)
formatter = logging.Formatter(
    fmt='[%(levelname)s] %(asctime)s [%(filename)12s:%(lineno)5d]:\t%(message)s'
)
console = logging.StreamHandler()
console.setFormatter(formatter)
log.addHandler(console)
log.setLevel(logging.DEBUG)

def gen_arr(data, dtype):
    num = len(data) // struct.calcsize(dtype)
    arr = struct.unpack('%d%s' % (num, dtype), data)
    return arr

def enc():
    with open(args.input_file, 'rb') as fin, open(args.output_file, 'wb') as fout:
        def read(fmt):
            data = fin.read(struct.calcsize(fmt))
            fout.write(data)
            return struct.unpack(fmt, data)
        _, = read('I')  # version
        lodsize, = read('Q')
        if lodsize != 0:
            log.warning('shit, it is LOD tensor!!! skipped!!')
            return None
        _, = read('I')  # version
        pbsize, = read('i')
        data = fin.read(pbsize)
        fout.write(data)
        proto = framework_pb2.VarType.TensorDesc()
        proto.ParseFromString(data)
        log.info('type: [%s] dim %s' % (proto.data_type, proto.dims))
        if proto.data_type == framework_pb2.VarType.FP32:
            arr = np.array(
                gen_arr(fin.read(), 'f'), dtype=np.float32).reshape(proto.dims)
            assert args.id is not None
            arr[-3, :] = np.ones_like(arr[0, :]) * args.id
            fout.write(arr.tobytes())
        else:
            raise RuntimeError('Unknown dtype %s' % proto.data_type)


def dec():
    with open(args.input_file, 'rb') as fin:
        def read(fmt):
            data = fin.read(struct.calcsize(fmt))
            return struct.unpack(fmt, data)
        _, = read('I')  # version
        lodsize, = read('Q')
        if lodsize != 0:
            log.warning('shit, it is LOD tensor!!! skipped!!')
            return None
        _, = read('I')  # version
        pbsize, = read('i')
        data = fin.read(pbsize)
        proto = framework_pb2.VarType.TensorDesc()
        proto.ParseFromString(data)
        if proto.data_type == framework_pb2.VarType.FP32:
            arr = np.array(
                gen_arr(fin.read(), 'f'), dtype=np.float32).reshape(proto.dims)
            log.info(arr[-3, :])
        else:
            raise RuntimeError('Unknown dtype %s' % proto.data_type)
        return arr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("-i", "--input_file", type=str, default=None)
    parser.add_argument("-o", "--output_file", type=str, default=None)
    parser.add_argument("-e", "--encrypt", action='store_true')
    parser.add_argument("-d", "--decrypt", action='store_true')
    parser.add_argument("--id", type=int, default=None)
    args = parser.parse_args()

    if args.encrypt:
        enc()
    elif args.decrypt:
        dec()
    else:
        log.error('do nothing')

