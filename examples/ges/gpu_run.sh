#!/bin/bash

export FLAGS_sync_nccl_allreduce=1
export FLAGS_eager_delete_tensor_gb=0
export FLAGS_fraction_of_gpu_memory_to_use=1
export NCCL_DEBUG=INFO
export NCCL_IB_GID_INDEX=3
export GLOG_v=1
export GLOG_logtostderr=1

num_nodes=10312
num_embedding=10351
num_sample_workers=20

# build train_data
rm -rf train_data && mkdir -p train_data 
cd train_data 
seq 0 $((num_nodes-1)) | shuf | split -l $((num_nodes/num_sample_workers+1))
cd - 

python3 gpu_train.py --output_path ./output  --epoch 100  --walk_len 40 --win_size 5 --neg_num 5 --batch_size 128 --hidden_size 128 \
    --num_nodes $num_nodes --num_embedding $num_embedding --num_sample_workers $num_sample_workers --steps_per_save 2000 --dataset "BlogCatalog"
