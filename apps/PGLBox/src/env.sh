#!/bin/bash
# environment variables for fleet distribute training
export FLAGS_LAUNCH_BARRIER=0
export PADDLE_TRAINERS=1
export PADDLE_TRAINERS_NUM=${PADDLE_TRAINERS}
export POD_IP=127.0.0.1
export PADDLE_PSERVERS_IP_PORT_LIST="127.0.0.1:29011"  #set free port if 29011 is occupied
export PADDLE_PSERVER_PORT_ARRAY=(29011)
export PADDLE_TRAINER_ID=0


export LD_PRELOAD=/root/pglbox/dependency/libjemalloc.so

# paddle
export FLAGS_call_stack_level=2
export FLAGS_use_stream_safe_cuda_allocator=false
export FLAGS_enable_opt_get_features=true
export FLAGS_gpugraph_enable_hbm_table_collision_stat=true
export FLAGS_gpugraph_hbm_table_load_factor=0.75
export FLAGS_gpugraph_enable_segment_merge_grads=true
export FLAGS_gpugraph_merge_grads_segment_size=128
export FLAGS_gpugraph_dedup_pull_push_mode=1
export FLAGS_gpugraph_load_node_list_into_hbm=false
export FLAGS_enable_exit_when_partial_worker=true

topo=`nvidia-smi topo -m | grep GPU0 | awk '{print $7}' | grep NV`
if [ "X${topo}" = "X" ]; then
    export FLAGS_gpugraph_enable_gpu_direct_access=false
    echo "FLAGS_gpugraph_enable_gpu_direct_access is false"
else
    export FLAGS_gpugraph_enable_gpu_direct_access=true
    echo "FLAGS_gpugraph_enable_gpu_direct_access is true"
fi


export FLAGS_graph_load_in_parallel=true
export FLAGS_graph_get_neighbor_id=false


export TRAINING_ROLE=TRAINER
export PADDLE_PORT=8800
