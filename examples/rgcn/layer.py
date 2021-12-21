#!usr/bin/env python
# -*- coding:utf-8 -*-
"""
/**************************************************************
 * Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
 * @author: sunzhuo02 
 * @email: sunzhuo02@baidu.com
 * @dept: KG
 * @file: layer.py
 * @time: 2021-12-06
 * @description: 
**************************************************************/
"""
import math
import numpy as np
import paddle.fluid as fluid

def RGCNConv(graph_wrapper, in_dim, out_dim, etypes, num_nodes, num_bases=0):
    """Implementation of Relational Graph Convolutional Networks (R-GCN)

    This is an implementation of the paper 
    Modeling Relational Data with Graph Convolutional Networks 
    (http://arxiv.org/abs/1703.06103).

    """
    param_attr_init = fluid.initializer.Uniform(
            low=-1.0, high=1.0, seed=np.random.randint(100))
    tn_initializer = fluid.initializer.TruncatedNormalInitializer(
            loc=0.0, scale=1.0 / math.sqrt(in_dim))
    
    def __message(src_feat, dst_feat, edge_feat):
        """
        send head信息
        """
        return src_feat["h"]
    
    def __reduce(msg):
        """
        sum pool
        """
        return fluid.layers.sequence_pool(msg, pool_type='sum')
    
    nfeat = fluid.layers.create_parameter(shape=[num_nodes, in_dim],
                                        default_initializer=param_attr_init,
                                        dtype='float32',
                                        name='nfeat')
    
    num_rels = len(etypes)
    if num_bases <= 0 or num_bases >= num_rels:
        num_bases = num_rels

    weight = fluid.layers.create_parameter(shape=[num_bases, in_dim, out_dim],
                                        default_initializer=tn_initializer,
                                        dtype='float32',
                                        name='trans_weight')

    if num_bases < num_rels:
        w_comp = fluid.layers.create_parameter(shape=[num_rels, num_bases],
                                        default_initializer=tn_initializer,
                                        dtype='float32',
                                        name='trans_w_comp')
        weight = fluid.layers.transpose(weight, perm=[1, 0, 2])
        weight = fluid.layers.matmul(w_comp, weight)
        # [num_rels, in_dim, out_dim]
        weight = fluid.layers.transpose(weight, perm=[1, 0, 2])

    feat_list = []
    for idx, etype in enumerate(etypes):
        w = weight[idx, :, :]

        h = fluid.layers.matmul(nfeat, w) 
        message = graph_wrapper[etype].send(__message, nfeat_list=[('h', h)])
        output = graph_wrapper[etype].recv(message, __reduce)
        feat_list.append(output)
        
    h = fluid.layers.sum(feat_list)
    return h
