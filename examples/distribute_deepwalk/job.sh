#!/bin/bash

set -x
source ./pgl_deepwalk.cfg

export CPU_NUM=$CPU_NUM
export FLAGS_rpc_deadline=3000000 
export FLAGS_communicator_send_queue_size=1
export FLAGS_communicator_min_send_grad_num_before_recv=0
export FLAGS_communicator_max_merge_var_num=1
export FLAGS_communicator_merge_sparse_grad=1

if [[ $build_train_data == True ]];then
    train_files="./train_data"
else
    train_files="None"
fi
    
if [[ $pre_walk == True ]]; then
    walkpath_files="./walk_path"
    if [[ $TRAINING_ROLE == "PSERVER" ]];then
        while [[ ! -d train_data ]];do
            sleep 60
            echo "Waiting for train_data ..."
        done
        rm -rf $walkpath_files && mkdir -p $walkpath_files
        python -u cluster_train.py --num_sample_workers $num_sample_workers --num_nodes $num_nodes --mode walk --walkpath_files $walkpath_files --epoch $epoch \
             --walk_len $walk_len --batch_size $batch_size --train_files $train_files --dataset "BlogCatalog"
        touch build_graph_done
    fi

    while [[ ! -f build_graph_done ]];do
        sleep 60
        echo "Waiting for walk_path ..."
    done
else
    walkpath_files="None"
fi

python -u cluster_train.py --num_sample_workers $num_sample_workers --num_nodes $num_nodes --optimizer $optimizer --walkpath_files $walkpath_files --epoch $epoch \
            --is_distributed $distributed_embedding --lr $learning_rate --neg_num $neg_num --walk_len $walk_len --win_size $win_size --is_sparse $is_sparse --hidden_size $dim \
            --batch_size $batch_size --steps_per_save $steps_per_save --train_files $train_files --dataset "BlogCatalog"
