# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import math
import math
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from sklearn.metrics import roc_auc_score
import torch
import paddle
import pgl
import numpy as np
import paddle.fluid as F
import paddle.fluid.layers as L

from pgl.contrib.ogb.nodeproppred.dataset_pgl import PglNodePropPredDataset
import time
import copy
from ogb.nodeproppred import Evaluator
import pickle as pkl

from model import Proteins_baseline_model, Proteins_label_embedding_model
from partition import RandomPartition as random_partition
from cross_edge_feat import cross_edge_feat
from dataset.random_partition_dataset import RandomPartition

import os
import argparse
from tqdm import tqdm

evaluator = Evaluator(name='ogbn-proteins')


def sample_subgraph(subgraph, max_neigh=64):
    nodes, eids = subgraph.sample_predecessor(
        nodes=subgraph.nodes,
        return_eids=True,
        shuffle=True,
        max_degree=max_neigh)

    eids = np.hstack(eids)
    new_edges = subgraph.edges[eids]
    new_edges_feat = subgraph.edge_feat["feat"][eids]
    subgraph = pgl.graph.Graph(
        num_nodes=subgraph.num_nodes,
        edges=new_edges,
        node_feat=subgraph.node_feat,
        edge_feat={"feat": new_edges_feat})
    return subgraph


def load_species():
    spid = pd.read_csv(
        "./dataset/ogbn_proteins_pgl/raw/species.csv.gz",
        header=None,
        names=["sid"])
    spids = pd.get_dummies(spid["sid"]).values
    spids = np.argmax(spids, 1)
    return spids


def get_config():
    parser = argparse.ArgumentParser()

    ## model_arg
    model_group = parser.add_argument_group('model_base_arg')
    model_group.add_argument('--num_layers', default=7, type=int)
    model_group.add_argument('--hidden_size', default=64, type=int)
    model_group.add_argument('--num_heads', default=4, type=int)
    model_group.add_argument('--dropout', default=0.1, type=float)
    model_group.add_argument('--attn_dropout', default=0, type=float)
    model_group.add_argument('--train_partition', default=9, type=int)
    model_group.add_argument('--eval_partition', default=3, type=int)
    model_group.add_argument('--cross_edge_feat', default=3, type=int)
    model_group.add_argument(
        '--cross_edge_feat_max_neigh', default=64, type=int)

    ## label_embed_arg
    embed_group = parser.add_argument_group('embed_arg')
    embed_group.add_argument('--use_label_e', action='store_true')
    embed_group.add_argument('--label_rate', default=0.5, type=float)

    ## train_arg
    train_group = parser.add_argument_group('train_arg')
    train_group.add_argument('--runs', default=10, type=int)
    train_group.add_argument('--epochs', default=5000, type=int)
    train_group.add_argument('--lr', default=0.001, type=float)
    train_group.add_argument('--place', default=-1, type=int)
    train_group.add_argument(
        '--log_file', default='result_proteins.txt', type=str)
    return parser.parse_args()


def optimizer_func(lr=0.01):
    return F.optimizer.AdamOptimizer(learning_rate=lr)


def eval_test(parser, program, model, test_exe, graph, y_true, split_idx,
              partition):

    y_pred = np.zeros_like(y_true)

    for subgraph, feed_dict in partition:
        batch_y_pred = test_exe.run(program=program,
                                    feed=feed_dict,
                                    fetch_list=[model.out_feat])[0]

        y_pred[subgraph.node_feat["nid"]] = batch_y_pred

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['rocauc']
    val_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['rocauc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['rocauc']

    return train_acc, val_acc, test_acc


def train_loop(parser,
               start_program,
               main_program,
               test_program,
               model,
               graph,
               label,
               split_idx,
               exe,
               run_id,
               wf=None):
    #build up training program
    exe.run(start_program)

    max_acc = 0  # best test_acc
    max_step = 0  # step for best test_acc 
    max_val_acc = 0  # best val_acc
    max_cor_acc = 0  # test_acc for best val_acc
    max_cor_step = 0  # step for best val_acc

    graph.node_feat["label"] = label
    graph.node_feat["nid"] = np.arange(0, graph.num_nodes)

    def train_batch_func(subgraph):
        if parser.use_label_e:
            train_idx = copy.deepcopy(split_idx['train'])
            np.random.shuffle(train_idx[:int(50125)])

            label_idx = train_idx[:int(50125 * parser.label_rate)]
            unlabel_idx = train_idx[int(50125 * parser.label_rate):]

            label_idx_total = set(label_idx)
            unlabel_idx_total = set(unlabel_idx)

            fd = model.gw.to_feed(subgraph)
            fd.update(model.gw_sample.to_feed(sample_subgraph(subgraph)))
            sub_idx = set(subgraph.node_feat["nid"])

            train_idx_temp = label_idx_total & sub_idx
            label_idx = subgraph.reindex_from_parrent_nodes(
                list(train_idx_temp))

            train_idx_temp = unlabel_idx_total & sub_idx
            unlabel_idx = subgraph.reindex_from_parrent_nodes(
                list(train_idx_temp))

            fd['label'] = subgraph.node_feat["label"]
            fd['label_idx'] = label_idx
            fd['train_idx'] = unlabel_idx
        else:
            fd = model.gw.to_feed(subgraph)
            fd.update(model.gw_sample.to_feed(sample_subgraph(subgraph)))
            train_idx_temp = set(split_idx['train']) & set(subgraph.node_feat[
                "nid"])
            train_idx_temp = subgraph.reindex_from_parrent_nodes(
                list(train_idx_temp))
            fd['label'] = subgraph.node_feat["label"]
            fd['train_idx'] = train_idx_temp
        return fd

    eval_partition = random_partition(
        num_clusters=parser.eval_partition, graph=graph, shuffle=False)
    train_partition = RandomPartition(
        batch_func=train_batch_func,
        graph=graph,
        num_cluster=parser.train_partition,
        epoch=parser.epochs,
        shuffle=True)

    def eval_func(subgraph):
        fd = model.gw.to_feed(subgraph)
        fd.update(model.gw_sample.to_feed(sample_subgraph(subgraph)))

        if parser.use_label_e:
            fd['label'] = subgraph.node_feat["label"]
            train_idx_temp = set(split_idx['train']) & set(subgraph.node_feat[
                "nid"])
            train_idx_temp = subgraph.reindex_from_parrent_nodes(
                list(train_idx_temp))
            fd['label_idx'] = train_idx_temp
        return fd

    eval_partition = [(subgraph, eval_func(subgraph))
                      for subgraph in eval_partition]

    epoch_id = 0
    batch_id = 0

    result = eval_test(parser, test_program, model, exe, graph, label,
                       split_idx, eval_partition)

    for feed_dict in tqdm(train_partition.generator()):
        #start training  
        batch_id += 1

        loss = exe.run(main_program,
                       feed=feed_dict,
                       fetch_list=[model.avg_cost])
        loss = loss[0]

        #eval result
        if batch_id % parser.train_partition == 0:
            epoch_id += 1

        if batch_id % 500 == 0:
            result = eval_test(parser, test_program, model, exe, graph, label,
                               split_idx, eval_partition)
            train_acc, valid_acc, test_acc = result

            max_acc = max(test_acc, max_acc)
            if max_acc == test_acc:
                max_step = epoch_id
            max_val_acc = max(valid_acc, max_val_acc)
            if max_val_acc == valid_acc:
                max_cor_acc = test_acc
                max_cor_step = epoch_id
            max_acc = max(result[2], max_acc)
            if max_acc == result[2]:
                max_step = epoch_id
            result_t = (f'Run: {run_id:02d}, '
                        f'Epoch: {epoch_id:02d}, '
                        f'Train: {100 * train_acc:.2f}%, '
                        f'Valid: {100 * valid_acc:.2f}%, '
                        f'Test: {100 * test_acc:.2f}% \n'
                        f'max_Test: {100 * max_acc:.2f}%, '
                        f'max_step: {max_step}\n'
                        f'max_val: {100 * max_val_acc:.2f}%, '
                        f'max_val_Test: {100 * max_cor_acc:.2f}%, '
                        f'max_val_step: {max_cor_step}\n')
            print(result_t)
            wf.write(result_t)
            wf.write('\n')
            wf.flush()
    return max_cor_acc


def np_scatter(idx, vals, target):
    """target[idx] += vals, but allowing for repeats in idx"""
    np.add.at(target, idx, vals)


def aggregate_node_features(graph):
    efeat = graph.edge_feat["feat"]
    graph.edge_feat["feat"] = efeat
    nfeat = np.zeros((graph.num_nodes, efeat.shape[-1]), dtype="float32")
    edges_dst = graph.edges[:, 1]
    np_scatter(edges_dst, efeat, nfeat)
    nfeat = nfeat / np.sqrt(graph.indegree()).reshape(-1, 1)
    graph.node_feat["feat"] = nfeat


if __name__ == '__main__':
    parser = get_config()
    print('===========args==============')
    print(parser)
    print('=============================')

    dataset = PglNodePropPredDataset(name="ogbn-proteins")
    split_idx = dataset.get_idx_split()

    graph, label = dataset[0]
    aggregate_node_features(graph)

    place = F.CPUPlace() if parser.place < 0 else F.CUDAPlace(parser.place)

    startup_prog = F.default_startup_program()
    train_prog = F.default_main_program()

    with F.program_guard(train_prog, startup_prog):
        with F.unique_name.guard():
            gw_sample = pgl.graph_wrapper.GraphWrapper(
                name="proteins_sampled",
                node_feat=graph.node_feat_info(),
                edge_feat=graph.edge_feat_info())

            gw = pgl.graph_wrapper.GraphWrapper(
                name="proteins",
                node_feat=graph.node_feat_info(),
                edge_feat=graph.edge_feat_info())

            old_feat = gw.node_feat["feat"]

            cross_feat = cross_edge_feat(
                gw_sample,
                gw_sample.edge_feat["feat"],
                parser.hidden_size,
                num_layers=parser.cross_edge_feat,
                max_neigh=parser.cross_edge_feat_max_neigh)

            if cross_feat is not None:
                new_node_feat = L.concat(
                    [gw.node_feat["feat"], cross_feat], axis=-1)
                gw.node_feat["feat"] = new_node_feat

            if parser.use_label_e:
                model = Proteins_label_embedding_model(
                    gw, parser.hidden_size, parser.num_heads, parser.dropout,
                    parser.num_layers)
            else:
                model = Proteins_baseline_model(
                    gw, parser.hidden_size, parser.num_heads, parser.dropout,
                    parser.num_layers)
            model.gw_sample = gw_sample
            model.gw.node_feat["feat"] = old_feat

            test_prog = train_prog.clone(for_test=True)
            model.train_program()

            # Recompute
            adam_optimizer = optimizer_func(parser.lr)
            adam_optimizer = F.optimizer.RecomputeOptimizer(adam_optimizer)
            adam_optimizer._set_checkpoints(model.checkpoints)
            adam_optimizer.minimize(model.avg_cost)

    exe = F.Executor(place)

    wf = open(parser.log_file, 'w', encoding='utf-8')
    total_test_acc = 0.0
    for run_i in range(parser.runs):
        total_test_acc += train_loop(parser, startup_prog, train_prog,
                                     test_prog, model, graph, label, split_idx,
                                     exe, run_i, wf)
    wf.write(f'average: {100 * (total_test_acc/parser.runs):.2f}%')
    wf.close()
