#!/bin/bash

set -x
source ./utils.sh

export CPU_NUM=$CPU_NUM
export FLAGS_rpc_deadline=3000000 

export FLAGS_communicator_send_queue_size=1
export FLAGS_communicator_min_send_grad_num_before_recv=0
export FLAGS_communicator_max_merge_var_num=1
export FLAGS_communicator_merge_sparse_grad=0

python -u cluster_train.py -c config.yaml
