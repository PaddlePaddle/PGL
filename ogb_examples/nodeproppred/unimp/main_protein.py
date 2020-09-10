import math
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

from utils import to_undirected, add_self_loop, linear_warmup_decay
from model import Proteins_baseline_model, Proteins_label_embedding_model
from partition import random_partition_v2 as random_partition

import argparse
from tqdm import tqdm
evaluator = Evaluator(name='ogbn-proteins')

# place=F.CUDAPlace(6)

def get_config():
    parser = argparse.ArgumentParser()
    
    ## model_arg
    model_group=parser.add_argument_group('model_base_arg')
    model_group.add_argument('--num_layers', default=7, type=int)
    model_group.add_argument('--hidden_size', default=64, type=int)
    model_group.add_argument('--num_heads', default=4, type=int)
    model_group.add_argument('--dropout', default=0.1, type=float)
    model_group.add_argument('--attn_dropout', default=0, type=float)
    
    ## label_embed_arg
    embed_group=parser.add_argument_group('embed_arg')
    embed_group.add_argument('--use_label_e', action='store_true')
    embed_group.add_argument('--label_rate', default=0.5, type=float)
    
    ## train_arg
    train_group=parser.add_argument_group('train_arg')
    train_group.add_argument('--runs', default=10, type=int )
    train_group.add_argument('--epochs', default=2000, type=int )
    train_group.add_argument('--lr', default=0.001, type=float)
    train_group.add_argument('--place', default=-1, type=int)
    train_group.add_argument('--log_file', default='result_proteins.txt', type=str)
    return parser.parse_args()

def optimizer_func(lr=0.01):
    return F.optimizer.AdamOptimizer(learning_rate=lr)


def eval_test(parser, program, model, test_exe, graph, y_true, split_idx):

    y_pred = np.zeros_like(y_true)

    graph.node_feat["label"] = y_true 
    graph.node_feat["nid"] = np.arange(0, graph.num_nodes) 
    for subgraph in random_partition(num_clusters=5, graph=graph, shuffle=False): 
        feed_dict = model.gw.to_feed(subgraph)
        if parser.use_label_e:
            feed_dict['label'] = subgraph.node_feat["label"]
            train_idx_temp = set(split_idx['train']) & set(subgraph.node_feat["nid"])
            train_idx_temp = subgraph.reindex_from_parrent_nodes(list(train_idx_temp))
            feed_dict['label_idx'] = train_idx_temp

        batch_y_pred = test_exe.run(
                program=program,
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

    
    

def train_loop(parser, start_program, main_program, test_program, 
               model, graph, label, split_idx, exe, run_id, wf=None):
    #build up training program
    exe.run(start_program)
    
    max_acc=0  # best test_acc
    max_step=0 # step for best test_acc 
    max_val_acc=0 # best val_acc
    max_cor_acc=0 # test_acc for best val_acc
    max_cor_step=0 # step for best val_acc
    #training loop

    graph.node_feat["label"] = label
    graph.node_feat["nid"] = np.arange(0, graph.num_nodes) 
    
    if parser.use_label_e:
        train_idx=copy.deepcopy(split_idx['train'])
        np.random.shuffle(train_idx[:50125])
        label_idx = train_idx[: int(50125*parser.label_rate)]
        unlabel_idx = train_idx[int(50125*parser.label_rate): ]
        label_idx_total= set(label_idx)
        unlabel_idx_total= set(unlabel_idx)   
    
    for epoch_id in tqdm(range(parser.epochs)):
        for subgraph in random_partition(num_clusters=9, graph=graph, shuffle=True): 
            #start training  
            if parser.use_label_e:
                feed_dict = model.gw.to_feed(subgraph)
                sub_idx = set(subgraph.node_feat["nid"])
                
                train_idx_temp = label_idx_total & sub_idx
                label_idx = subgraph.reindex_from_parrent_nodes(list(train_idx_temp))
                
                train_idx_temp = unlabel_idx_total & sub_idx
                unlabel_idx = subgraph.reindex_from_parrent_nodes(list(train_idx_temp))
                
                feed_dict['label'] = subgraph.node_feat["label"] 
                feed_dict['label_idx'] = label_idx
                feed_dict['train_idx'] = unlabel_idx
            else:
                feed_dict = model.gw.to_feed(subgraph)
                #feed_dict['label'] = label
                train_idx_temp = set(split_idx['train']) & set(subgraph.node_feat["nid"])
                train_idx_temp = subgraph.reindex_from_parrent_nodes(list(train_idx_temp))
                feed_dict['label'] = subgraph.node_feat["label"] 
                feed_dict['train_idx'] = train_idx_temp
            
            loss = exe.run(main_program,
                          feed=feed_dict,
                          fetch_list=[model.avg_cost])
            loss = loss[0]

        #eval result
        if (epoch_id+1) > parser.epochs*0.9:
            result = eval_test(parser, test_program, model, exe, graph, label, split_idx)
            train_acc, valid_acc, test_acc = result

            max_acc = max(test_acc, max_acc)
            if max_acc == test_acc:
                max_step=epoch_id
            max_val_acc=max(valid_acc, max_val_acc)
            if max_val_acc==valid_acc:
                max_cor_acc=test_acc
                max_cor_step=epoch_id
            max_acc=max(result[2], max_acc)
            if max_acc==result[2]:
                max_step=epoch_id
            result_t=(f'Run: {run_id:02d}, '
                      f'Epoch: {epoch_id:02d}, '
                      #f'Loss: {loss[0]:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}%, '
                      f'Test: {100 * test_acc:.2f}% \n'
                      f'max_Test: {100 * max_acc:.2f}%, '
                      f'max_step: {max_step}\n'
                      f'max_val: {100 * max_val_acc:.2f}%, '
                      f'max_val_Test: {100 * max_cor_acc:.2f}%, '
                      f'max_val_step: {max_cor_step}\n'
                     )
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
    graph.node_feat["feat"] = nfeat
    


if __name__ == '__main__':
    parser = get_config()
    print('===========args==============')
    print(parser)
    print('=============================')
    
    dataset = PglNodePropPredDataset(name="ogbn-proteins")
    split_idx=dataset.get_idx_split()
    
    graph, label = dataset[0]
    aggregate_node_features(graph)
    
    place=F.CPUPlace() if parser.place <0 else F.CUDAPlace(parser.place)
    
    startup_prog = F.default_startup_program()
    train_prog = F.default_main_program()
    
    with F.program_guard(train_prog, startup_prog):
        with F.unique_name.guard():
            gw = pgl.graph_wrapper.GraphWrapper(
                    name="proteins",
                    node_feat=graph.node_feat_info(),
                    edge_feat=graph.edge_feat_info())
            
            if parser.use_label_e:
                model = Proteins_label_embedding_model(gw, parser.hidden_size, parser.num_heads, 
                                                       parser.dropout, parser.num_layers)
            else:
                model = Proteins_baseline_model(gw, parser.hidden_size, parser.num_heads, 
                                                 parser.dropout, parser.num_layers)
                
            test_prog=train_prog.clone(for_test=True)
            model.train_program()
            
        
            adam_optimizer = optimizer_func(parser.lr)#optimizer
            adam_optimizer.minimize(model.avg_cost)
    
    exe = F.Executor(place)
    
    wf = open(parser.log_file, 'w', encoding='utf-8')
    total_test_acc=0.0
    for run_i in range(parser.runs):
        total_test_acc+=train_loop(parser, startup_prog, train_prog, test_prog, model,
            graph, label, split_idx, exe, run_i, wf)
    wf.write(f'average: {100 * (total_test_acc/parser.runs):.2f}%')
    wf.close()
