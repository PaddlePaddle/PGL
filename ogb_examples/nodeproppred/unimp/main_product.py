import math
import torch
import paddle
import pgl
import numpy as np
import paddle.fluid as F
import paddle.fluid.layers as L
import copy
from pgl.contrib.ogb.nodeproppred.dataset_pgl import PglNodePropPredDataset
from ogb.nodeproppred import Evaluator

from utils import to_undirected, add_self_loop, linear_warmup_decay
from model import Products_label_embedding_model
from dataloader.ogb_products_dataloader import SampleDataGenerator
import paddle.fluid.profiler as profiler
from pgl.utils import paddle_helper

import argparse
from tqdm import tqdm
evaluator = Evaluator(name='ogbn-products')

def get_config():
    parser = argparse.ArgumentParser()
    
    ## data_sampling_arg
    data_group= parser.add_argument_group('data_arg')
    data_group.add_argument('--batch_size', default=1500, type=int)
    data_group.add_argument('--num_workers', default=12, type=int)
    data_group.add_argument('--sizes', default=[10, 10, 10], type=int, nargs='+' )
    data_group.add_argument('--buf_size', default=1000, type=int)

    ## model_arg
    model_group=parser.add_argument_group('model_base_arg')
    model_group.add_argument('--num_layers', default=3, type=int)
    model_group.add_argument('--hidden_size', default=128, type=int)
    model_group.add_argument('--num_heads', default=4, type=int)
    model_group.add_argument('--dropout', default=0.3, type=float)
    model_group.add_argument('--attn_dropout', default=0, type=float)
    
    ## label_embed_arg
    embed_group=parser.add_argument_group('embed_arg')
    embed_group.add_argument('--use_label_e', action='store_true')
    embed_group.add_argument('--label_rate', default=0.625, type=float)
    
    ## train_arg
    train_group=parser.add_argument_group('train_arg')
    train_group.add_argument('--runs', default=10, type=int )
    train_group.add_argument('--epochs', default=100, type=int )
    train_group.add_argument('--lr', default=0.001, type=float)
    train_group.add_argument('--place', default=-1, type=int)
    train_group.add_argument('--log_file', default='result_products.txt', type=str)
    return parser.parse_args()


def optimizer_func(lr):
    return F.optimizer.AdamOptimizer(learning_rate=lr)


def eval_test(parser, test_p_list, model, test_exe, dataset, split_idx):
    
    eval_gg=SampleDataGenerator(graph_wrappers=[model.gw_list[0]], buf_size=parser.buf_size,
                                 batch_size=parser.batch_size , num_workers=1,
                                 sizes=[-1,], shuffle=False,
                                  dataset=dataset,
                                  nodes_idx=None)
    
    out_r_temp=[]
    test_p, out=test_p_list[0]
    
    pbar = tqdm(total=eval_gg.num_nodes* model.num_layers)
    pbar.set_description('Evaluating')
    
    for feed_batch in tqdm(eval_gg.generator()):
        feed_batch['label_idx']=split_idx['train']
        feat_batch= test_exe.run(test_p,
                              feed=feed_batch,
                              fetch_list=out)
        out_r_temp.append(feat_batch[0])
        pbar.update(feed_batch['label'].shape[0])
        
    our_r=np.concatenate(out_r_temp, axis=0)
     
    for test_p, out in test_p_list[1:]: #np.concatenate
        out_r_temp=[]
        for feed_batch in tqdm(eval_gg.generator()):

            feed_batch['hidden_node_feat'] = our_r[feed_batch['batch_nodes_0']]
            feat_batch= test_exe.run(test_p,
                                  feed=feed_batch,
                                  fetch_list=out)
            out_r_temp.append(feat_batch[0])
            pbar.update(feed_batch['label'].shape[0])
        our_r=np.concatenate(out_r_temp, axis=0)
    pbar.close()

    y_pred=our_r.argmax(axis=-1)
    y_pred=np.expand_dims(y_pred, 1)
    y_true=eval_gg.labels
    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    val_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, val_acc, test_acc

def train_loop(parser, start_program, main_program, test_p_list,
               model, feat_init, place, dataset, split_idx, exe, run_id, wf=None):
    #build up training program
    exe.run(start_program)
    feat_init(place)
    
    max_acc=0  # best test_acc
    max_step=0 # step for best test_acc 
    max_val_acc=0 # best val_acc
    max_cor_acc=0 # test_acc for best val_acc
    max_cor_step=0 # step for best val_acc
    #training loop

    for epoch_id in range(parser.epochs):
        #start training  
         
        if parser.use_label_e:
            train_idx_temp=copy.deepcopy(split_idx['train'])
            np.random.shuffle(train_idx_temp)
            label_idx=train_idx_temp[ :int(parser.label_rate*len(train_idx_temp))]
            unlabel_idx=train_idx_temp[int(parser.label_rate*len(train_idx_temp)):]
            train_gg=SampleDataGenerator(graph_wrappers=model.gw_list, buf_size=parser.buf_size,
                                 batch_size=parser.batch_size , num_workers=parser.num_workers,
                                 sizes=parser.sizes, shuffle=True,
                                  dataset=dataset,
                                  nodes_idx=unlabel_idx)
            pbar = tqdm(total=unlabel_idx.shape[0])
            pbar.set_description(f'Epoch {epoch_id:02d}')

            total=0.0
            acc_num=0.0
            for batch_feed in tqdm(train_gg.generator()):    

                batch_feed['label_idx']=label_idx
                loss = exe.run(main_program,
                          feed=batch_feed,
                          fetch_list=[model.avg_cost, model.out_feat])
                total+=loss[0][0]
                
                
                acc_num=(loss[1].argmax(axis=-1)==batch_feed['label'].reshape(-1)).sum()+acc_num
                pbar.update(batch_feed['label'].shape[0])
            pbar.close()
            print(total/(len(train_gg)/parser.batch_size))  

            print('acc: ', (acc_num/unlabel_idx.shape[0])*100)

        #eval result
        if (epoch_id+1)>=50 and (epoch_id+1)%10==0:
            result = eval_test(parser, test_p_list, model, exe, dataset, split_idx)
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
                      f'Loss: {total:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}%, '
                      f'Test: {100 * test_acc:.2f}% \n'
                      f'max_Test: {100 * max_acc:.2f}%, '
                      f'max_step: {max_step}\n'
                      f'max_val: {100 * max_val_acc:.2f}%, '
                      f'max_val_Test: {100 * max_cor_acc:.2f}%, '
                      f'max_val_step: {max_cor_step}\n'
                     )
#         if (epoch_id+1)%50==0:
            print(result_t)
            wf.write(result_t)
            wf.write('\n')
            wf.flush()
    return max_cor_acc





if __name__ == '__main__':
    parser = get_config()
    print('===========args==============')
    print(parser)
    print('=============================')
    
    startup_prog = F.default_startup_program()
    train_prog = F.default_main_program()

    
    place=F.CPUPlace() if parser.place <0 else F.CUDAPlace(parser.place)
    
    dataset = PglNodePropPredDataset(name="ogbn-products")
#     dataset = PglNodePropPredDataset(name="ogbn-arxiv")

    split_idx=dataset.get_idx_split()
    
    graph, label = dataset[0]
    print(label.shape)
    
    with F.program_guard(train_prog, startup_prog):
        with F.unique_name.guard():
            
            gw_list=[]
            
            for i in range(len(parser.sizes)):
                gw_list.append(pgl.graph_wrapper.GraphWrapper(
                    name="product_"+str(i)))

            feature_input, feat_init=paddle_helper.constant(
                    name='node_feat_input',
                    dtype='float32',
                    value=graph.node_feat['feat'])
    
            if parser.use_label_e:
                model=Products_label_embedding_model(feature_input, gw_list, 
                                                     parser.hidden_size, parser.num_heads, 
                                                        parser.dropout, parser.num_layers)
            else:
                model=Arxiv_baseline_model(gw, parser.hidden_size, parser.num_heads, 
                                                 parser.dropout, parser.num_layers)
                
#             test_prog=train_prog.clone(for_test=True)
            model.train_program()
           
            adam_optimizer = optimizer_func(parser.lr)#optimizer
            adam_optimizer.minimize(model.avg_cost)
    
    test_p_list=[] 
    
    with F.unique_name.guard():  

        ## build up eval program
        test_p=F.Program()
        with F.program_guard(test_p, ):
            gw_test=pgl.graph_wrapper.GraphWrapper(
                    name="product_"+str(0))

            feature_input, feat_init__=paddle_helper.constant(
                    name='node_feat_input',
                    dtype='float32',
                    value=graph.node_feat['feat'])
            label_feature=model.label_embed_input(model.feature_input)
            feature_batch=model.get_batch_feature(label_feature)  # 把batch_feat打出来

            feature_batch=model.get_gat_layer(0, gw_test, feature_batch, 
                                                 hidden_size=model.hidden_size,
                                             num_heads=model.num_heads, 
                                                  concat=True, 
                                             layer_norm=True, relu=True)
            sub_node_index=F.data(name='sub_node_index_0', shape=[None], 
                                  dtype="int64")
            feature_batch=L.gather(feature_batch, sub_node_index, overwrite=False)
#             test_p=test_p.clone(for_test=True)
            test_p_list.append((test_p, feature_batch))
            
        for i in range(1,model.num_layers-1):
            test_p=F.Program()
            with F.program_guard(test_p, ):
                gw_test=pgl.graph_wrapper.GraphWrapper(
                    name="product_"+str(0))
#                 feature_batch=model.get_batch_feature(label_feature, test=True) 
                feature_batch = F.data( 'hidden_node_feat',
                                    shape=[None, model.num_heads*model.hidden_size],
                                    dtype='float32')   
                feature_batch=model.get_gat_layer(i, gw_test, feature_batch, 
                                                 hidden_size=model.hidden_size,
                                             num_heads=model.num_heads,
                                                  concat=True, 
                                             layer_norm=True, relu=True)
                sub_node_index=F.data(name='sub_node_index_0', shape=[None], 
                                      dtype="int64")
                feature_batch=L.gather(feature_batch, sub_node_index, overwrite=False)    
#                 test_p=test_p.clone(for_test=True)
                test_p_list.append((test_p, feature_batch))
            
        test_p=F.Program()
        with F.program_guard(test_p, ):
            gw_test=pgl.graph_wrapper.GraphWrapper(
                    name="product_"+str(0))
#             feature_batch=model.get_batch_feature(label_feature, test=True)
            feature_batch = F.data( 'hidden_node_feat',
                                    shape=[None, model.num_heads*model.hidden_size ],
                                    dtype='float32')
            feature_batch = model.get_gat_layer(model.num_layers-1, gw_test, feature_batch, 
                                           hidden_size=model.out_size, num_heads=model.num_heads, 
                                             concat=False, layer_norm=False, relu=False, gate=True)
            sub_node_index=F.data(name='sub_node_index_0', shape=[None], 
                                  dtype="int64")
            feature_batch=L.gather(feature_batch, sub_node_index, overwrite=False)
#             test_p=test_p.clone(for_test=True)
            test_p_list.append((test_p, feature_batch))    
    
    
    exe = F.Executor(place)
    
    wf = open(parser.log_file, 'w', encoding='utf-8')
    total_test_acc=0.0
    for run_i in range(parser.runs):
        total_test_acc+=train_loop(parser, startup_prog, train_prog, test_p_list, model, feat_init,
            place, dataset, split_idx, exe, run_i, wf)
    wf.write(f'average: {100 * (total_test_acc/parser.runs):.2f}%')
    wf.close()
