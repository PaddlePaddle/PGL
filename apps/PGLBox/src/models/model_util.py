#-*- coding: utf-8 -*-
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
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
"""
    model utils
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
sys.path.append("../")
import math
import time
import numpy as np
import pickle as pkl

import paddle
import paddle.fluid as F
import paddle.fluid.layers as L
import pgl
from pgl.utils.logger import log

from models import layers 
import util
import helper

def inner_add(value, var):
    """ inner add """
    tmp = var + value
    L.assign(tmp, var)

def calc_auc(pos_logits, neg_logits):
    """calc_auc"""
    pos_logits = L.reshape(pos_logits[:, 0], [-1, 1])
    neg_logits = L.reshape(neg_logits[:, 0], [-1, 1])
    proba = L.concat([pos_logits, neg_logits], 0)
    proba = L.concat([proba * -1 + 1, proba], axis=1)

    pos_labels = L.ones_like(pos_logits)
    neg_labels = L.zeros_like(neg_logits)

    pos_labels = L.cast(pos_labels, dtype="int64")
    neg_labels = L.cast(neg_labels, dtype="int64")

    labels = L.concat([pos_labels, neg_labels], 0)
    labels.stop_gradient=True
    _, batch_auc_out, state_tuple = L.auc(input=proba, label=labels, num_thresholds=4096)
    return batch_auc_out, state_tuple

def dump_func(file_obj, node_index, node_embed):
    """paddle dump embedding

    输入:

        file_obj: 文件锁对象
        node_index: 节点ID
        node_embed: 节点embedding
        写出路径： output_path/part-%s worker-index
        写出格式: index \t label1 \t label1_score \t label2 \t label2_score

    """
    out = F.default_main_program().current_block().create_var(
                name="dump_out", dtype="float32", shape=[1])
    def _dump_func(vec_id, node_vec):
        """
            Dump Vectors for inference
        """
        if True:
            buffer
            file_obj.acquire()
            vec_lines = []
            for _node_id, _vec_feat in zip(np.array(vec_id), np.array(node_vec)):
                _node_id = str(_node_id.astype("int64").astype('uint64')[0])
                _vec_feat = " ".join(["%.5lf" % w for w in _vec_feat])

                vec_lines.append("%s\t%s\n" % (_node_id, _vec_feat))

            if len(vec_lines) > 0:
                file_obj.vec_path.write(''.join(vec_lines))
                vec_lines = []
            file_obj.release()
        return np.array([1], dtype="float32")

    o = L.py_func(_dump_func, [node_index, node_embed], out=out)
    return o

def paddle_print(*args):
    """print auc"""
    global print_count
    print_count = 0

    global start_time
    start_time = time.time()


    def _print(*inputs):
        """print auc by batch"""
        global print_count
        global start_time
        print_count += 1
        print_per_step = 1000
        if print_count % print_per_step == 0:
            speed = 1.0 * (time.time() - start_time) / print_per_step
            msg = "Speed %s sec/batch \t Batch:%s\t " % (speed, print_count)
            for x in inputs:
                msg += " %s \t" % (np.array(x)[0])
            log.info(msg)
            start_time = time.time()

    L.py_func(_print, args, out=None)

def loss_visualize(loss):
    """loss_visualize"""
    visualize_loss = L.create_global_var(
            persistable=True, dtype="float32", shape=[1], value=0)
    batch_count = L.create_global_var(
            persistable=True, dtype="float32", shape=[1], value=0)
    inner_add(loss, visualize_loss)
    inner_add(1., batch_count)

    return visualize_loss, batch_count

def build_node_holder(nodeid_slot_name):
    """ build node holder """
    holder_list = []
    nodeid_slot_holder = L.data(str(nodeid_slot_name), 
                                shape=[-1, 1], 
                                dtype="int64", 
                                lod_level=1)

    show = L.data("show", shape=[-1], dtype="int64")
    click = L.data("click", shape=[-1], dtype="int64")
    show_clk = L.concat([L.reshape(show, [-1, 1]), L.reshape(click, [-1, 1])], axis=-1)
    show_clk = L.cast(show_clk, dtype="float32")
    show_clk.stop_gradient = True
    holder_list = [nodeid_slot_holder, show, click]

    return nodeid_slot_holder, show_clk, holder_list

def build_slot_holder(discrete_slot_names):
    """build discrete slot holders """
    holder_list = []
    discrete_slot_holders = []
    discrete_slot_lod_holders = []
    for slot in discrete_slot_names:
        holder = L.data(slot,
                        shape=[None, 1],
                        dtype="int64",
                        lod_level=1,
                        append_batch_size=False)
        discrete_slot_holders.append(holder)
        holder_list.append(holder)

        lod_holder = L.data("slot_%s_lod" % slot,
                            shape=[None],
                            dtype="int64",
                            lod_level=0,
                            append_batch_size=False)
        discrete_slot_lod_holders.append(lod_holder)
        holder_list.append(lod_holder)

    return discrete_slot_holders, discrete_slot_lod_holders, holder_list

def build_token_holder(token_slot_name):
    """build token slot holder """
    token_slot_name = str(token_slot_name)
    token_slot_holder = L.data(token_slot_name,
                                shape=[None, 1],
                                dtype="int64",
                                lod_level=1,
                                append_batch_size=False)

    token_slot_lod_holder = L.data("slot_%s_lod" % token_slot_name,
                                shape=[None],
                                dtype="int64",
                                lod_level=0,
                                append_batch_size=False)

    return token_slot_holder, token_slot_lod_holder

def build_graph_holder(samples):
    """ build graph holder """
    holder_list = []
    graph_holders = {}
    for i, s in enumerate(samples):
        # For different sample size, we hold a graph block.
        graph_holders[i] = []
        num_nodes = L.data(
            name="%s_num_nodes" % i,
            shape=[-1],
            dtype="int")
        graph_holders[i].append(num_nodes)
        holder_list.append(num_nodes)

        next_num_nodes = L.data(
            name="%s_next_num_nodes" % i,
            shape=[-1],
            dtype="int")
        graph_holders[i].append(next_num_nodes)
        holder_list.append(next_num_nodes)

        edges_src = L.data(
            name="%s_edges_src" % i,
            shape=[-1, 1],
            dtype="int64")
        graph_holders[i].append(edges_src)
        holder_list.append(edges_src)

        edges_dst = L.data(
            name="%s_edges_dst" % i,
            shape=[-1, 1],
            dtype="int64")
        graph_holders[i].append(edges_dst)
        holder_list.append(edges_dst)

        edges_split = L.data(
            name="%s_edges_split" % i,
            shape=[-1],
            dtype="int")
        graph_holders[i].append(edges_split)
        holder_list.append(edges_split)

    ego_index_holder = L.data(name="final_index", shape=[-1], dtype="int")
    holder_list.append(ego_index_holder)

    return graph_holders, ego_index_holder, holder_list

def get_sparse_embedding(config,
        nodeid_slot_holder,
        discrete_slot_holders,
        discrete_slot_lod_holders,
        show_clk,
        use_cvm,
        emb_size,
        name="embedding"):
    """get sparse embedding"""

    id_embedding = F.contrib.sparse_embedding(input=nodeid_slot_holder,
                                size=[1024, emb_size + 3],
                                param_attr=F.ParamAttr(name=name))

    id_embedding = L.continuous_value_model(id_embedding, show_clk, use_cvm)
    id_embedding = id_embedding[:, 1:] # the first column is for lr, remove it

    tmp_slot_emb_list = []
    for slot_idx, lod in zip(discrete_slot_holders, discrete_slot_lod_holders):
        slot_emb = F.contrib.sparse_embedding(input=slot_idx,
                                    size=[1024, emb_size + 3],
                                    param_attr=F.ParamAttr(name=name))

        lod = L.cast(lod, dtype="int32")
        lod = L.reshape(lod, [1, -1])
        lod.stop_gradient = True
        slot_emb = L.lod_reset(slot_emb, lod)

        tmp_slot_emb_list.append(slot_emb)

    slot_embedding_list = []
    if(len(discrete_slot_holders)) > 0:
        slot_bows = F.contrib.layers.fused_seqpool_cvm(
                tmp_slot_emb_list, 
                config.slot_pool_type,
                show_clk,
                use_cvm=use_cvm)
        for bow in slot_bows:
            slot_embedding = bow[:, 1:]
            slot_embedding = L.softsign(slot_embedding)
            slot_embedding_list.append(slot_embedding)

    return id_embedding, slot_embedding_list

def get_layer(layer_type, graph, feature, next_num_nodes, hidden_size, act, name, is_test=False):
    """get_layer"""
    return getattr(layers, layer_type)(graph, feature, next_num_nodes, hidden_size, act, name)

def gnn_layers(graph_holders, init_feature, hidden_size, layer_type, \
    act, num_layers, etype_len, alpha_residual, interact_mode="sum"):
    """ generate graphsage layers"""

    # Pad 0 feature to deal with empty edges, otherwise will raise errors.
    zeros_tensor1 = paddle.zeros([1, init_feature.shape[-1]])
    zeros_tensor2 = paddle.zeros([1, 1], dtype="int64")
    init_feature = paddle.concat([zeros_tensor1, init_feature])
    feature = init_feature
    
    for i in range(num_layers):
        if i == num_layers - 1:
            act = None
        graph_holder = graph_holders[num_layers - i - 1]
        num_nodes = graph_holder[0] + 1
        next_num_nodes = graph_holder[1] + 1
        edges_src = graph_holder[2] + 1
        edges_dst = graph_holder[3] + 1
        split_edges = paddle.cumsum(graph_holder[4])
        nxt_fs = []
        for j in range(etype_len):
            start = paddle.zeros([1], dtype="int64") if j == 0 else split_edges[j - 1]
            new_edges_src = paddle.concat(
                [zeros_tensor2, edges_src[start: split_edges[j]]])
            new_edges_dst = paddle.concat(
                [zeros_tensor2, edges_dst[start: split_edges[j]]])
            graph = pgl.Graph(num_nodes=num_nodes,
                              edges=paddle.concat([new_edges_src, new_edges_dst], axis=1))
            nxt_f = get_layer(
                layer_type,
                graph,
                feature,
                next_num_nodes,
                hidden_size,
                act,
                name="%s_%s_%s" % (layer_type, i, j))
            nxt_fs.append(nxt_f)

        feature = feature_interact_by_etype(
            nxt_fs, interact_mode, hidden_size, name="%s_interat" % layer_type)
        feature = init_feature[:next_num_nodes] * alpha_residual + feature * (1 - alpha_residual)
    return feature[1:]

def feature_interact_by_etype(feature_list, interact_mode, hidden_size, name):
    """ feature interact with etype """
    if len(feature_list) == 1:
        return feature_list[0]
    elif interact_mode == "gatne":
        # stack [num_nodes, num_etype, hidden_size]
        U = L.stack(feature_list, axis=1)
        #  [num_nodes * num_etype, hidden_size]
        tmp = L.fc(L.reshape(U, shape=[-1, hidden_size]),
                hidden_size,
                act="tanh",
                param_attr=F.ParamAttr(name=name + '_w1'),
                bias_attr=None)

        #  [num_nodes * num_etype, 1]
        tmp = L.fc(tmp, 1, act=None, param_attr=F.ParamAttr(name=name + '_w2'))
        #  [num_nodes, num_etype]
        tmp = L.reshape(tmp, shape=[-1, len(feature_list)])
        #  [num_nodes, 1, num_etype]
        a = L.unsqueeze(L.softmax(tmp, axis=-1), axes=1)
        out = L.squeeze(L.matmul(a, U), axes=[1])
        return out
    else:
        return L.sum(feature_list)

def dump_embedding(config, nfeat, node_index):
    """dump_embedding"""
    node_embed = L.squeeze(nfeat, axes=[1], name=config.dump_node_emb_name)
    node_index = L.reshape(node_index, shape=[-1, 2])
    src_node_index = node_index[:, 0:1]
    src_node_index = L.reshape(src_node_index, 
            shape = src_node_index.shape, 
            name=config.dump_node_name) # for rename

def hcl(config, feature, graph_holders):
    """Hierarchical Contrastive Learning"""
    hcl_logits = []
    for idx, sample in enumerate(config.samples):
        graph_holder = graph_holders[idx]
        edges_src = graph_holder[2]
        edges_dst = graph_holder[3]
        neighbor_src_feat = paddle.gather(feature, edges_src)
        neighbor_src_feat = neighbor_src_feat.reshape([-1, 1, config.emb_size])
        neighbor_dst_feat = paddle.gather(feature, edges_dst)
        neighbor_dst_feat = neighbor_dst_feat.reshape([-1, 1, config.emb_size])
        neighbor_dsts_feat_all = [neighbor_dst_feat]

        for neg in range(config.neg_num):
            neighbor_dsts_feat_all.append(
                    F.contrib.layers.shuffle_batch(neighbor_dsts_feat_all[0]))
        neighbor_dsts_feat = L.concat(neighbor_dsts_feat_all, axis=1)

        # [batch_size, 1, neg_num+1]
        logits = L.matmul(neighbor_src_feat, neighbor_dsts_feat, transpose_y=True)  
        logits = L.squeeze(logits, axes=[1])
        hcl_logits.append(logits)

    return hcl_logits

def reset_program_state_dict(args, model, state_dict, pretrained_state_dict):
    """
    Initialize the parameter from the bert config, and set the parameter by 
    reseting the state dict."
    """
    scale = args.init_range
    reset_state_dict = {}
    reset_parameter_names = []
    for n, p in state_dict.items():
        if n in pretrained_state_dict:
            log.info("p_name: %s , pretrained name: %s" % (p.name, n))
            reset_state_dict[p.name] = np.array(pretrained_state_dict[n])
            reset_parameter_names.append(n)
        #  elif p.name in pretrained_state_dict and "bert" in n:
        #      reset_state_dict[p.name] = np.array(pretrained_state_dict[p.name])
        #      reset_parameter_names.append(n)
        else:
            log.info("[RANDOM] p_name: %s , pretrained name: %s" % (p.name, n))
            dtype_str = "float32"
            if str(p.dtype) == "VarType.FP64":
                dtype_str = "float64"
            reset_state_dict[p.name] = np.random.normal(
                loc=0.0, scale=scale, size=p.shape).astype(dtype_str)
    log.info("the following parameter had reset, please check. {}".format(
        reset_parameter_names))
    return reset_state_dict


