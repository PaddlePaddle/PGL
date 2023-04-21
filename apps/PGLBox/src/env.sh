export GLOG_v=-1

export FLAGS_LAUNCH_BARRIER=0
export PADDLE_TRAINERS=1
export FLAGS_enable_tracker_all2all=false
export FLAGS_enable_auto_rdma_trans=true
export FLAGS_enable_all2all_use_fp16=true
export FLAGS_enable_sparse_inner_gather=true
export FLAGS_check_nan_inf=false

if [[ ! -z "$MPI_NODE_NUM" ]] && [[ $MPI_NODE_NUM -gt 1 ]]; then
    echo "PADDLE_TRAINER_ID: $PADDLE_TRAINER_ID, PADDLE_TRAINER_ENDPOINTS: $PADDLE_TRAINER_ENDPOINTS, PADDLE_CURRENT_ENDPOINT: $PADDLE_CURRENT_ENDPOINT"
    export PADDLE_WITH_GLOO=2
    export PADDLE_GLOO_RENDEZVOUS=3
    #export PADDLE_GLOO_HTTP_ENDPOINT=${PADDLE_TRAINER_ENDPOINTS/,*/}
else
    export PADDLE_TRAINERS_NUM=${PADDLE_TRAINERS}
    export POD_IP=127.0.0.1
    export PADDLE_PSERVERS_IP_PORT_LIST="127.0.0.1:29011"  #set free port if 29011 is occupied
    export PADDLE_TRAINER_ENDPOINTS=${PADDLE_TRAINER_ENDPOINTS/,*/}
    export PADDLE_PSERVER_PORT_ARRAY=(29011)
    export PADDLE_TRAINER_ID=0
    export TRAINING_ROLE=TRAINER
    export PADDLE_PORT=8800
fi

export LD_PRELOAD=/pglbox_dependency/libjemalloc.so

export FLAGS_call_stack_level=2
export FLAGS_free_when_no_cache_hit=true
export FLAGS_use_stream_safe_cuda_allocator=true
export FLAGS_gpugraph_enable_hbm_table_collision_stat=false
export FLAGS_gpugraph_hbm_table_load_factor=0.75
export FLAGS_gpugraph_enable_segment_merge_grads=true
export FLAGS_gpugraph_merge_grads_segment_size=128
export FLAGS_gpugraph_dedup_pull_push_mode=1
export FLAGS_gpugraph_load_node_list_into_hbm=false
export FLAGS_enable_exit_when_partial_worker=true
export FLAGS_gpugraph_debug_gpu_memory=false
export FLAGS_gpugraph_slot_feasign_max_num=200
export FLAGS_gpugraph_enable_gpu_direct_access=false
export FLAGS_graph_load_in_parallel=true
export FLAGS_graph_get_neighbor_id=false

# storage mode
# 1. WHOLE_HBM 2.MEM_EMBEDDING_NO_FEATURE"(currently not supported)
# 3.MEM_EMBEDDING 4.SSD_EMBEDDING
train_storage_mode=`grep train_storage_mode ./config.yaml | sed s/#.*//g | grep train_storage_mode | awk -F':' '{print $2}' | sed 's/ //g'`
if [ "${train_storage_mode}" = "WHOLE_HBM" ]; then
    export FLAGS_gpugraph_storage_mode=1
    echo "FLAGS_gpugraph_storage_mode is WHOLE_HBM"
elif [ "${train_storage_mode}" = "SSD_EMBEDDING" ]; then
    export FLAGS_gpugraph_storage_mode=4
    echo "FLAGS_gpugraph_storage_mode is SSD_EMBEDDING"
else
    export FLAGS_gpugraph_storage_mode=3
    echo "FLAGS_gpugraph_storage_mode is MEM_EMBEDDING"
fi

metapath_split_opt=`grep metapath_split_opt ./config.yaml | sed s/#.*//g | grep metapath_split_opt | awk -F':' '{print $2}' | sed 's/ //g'`
if [ "${metapath_split_opt}" == "True" ] || [ "${metapath_split_opt}" == "true" ];then
    export FLAGS_graph_metapath_split_opt=true
    echo "FLAGS_graph_metapath_split_opt is true"
else
    export FLAGS_graph_metapath_split_opt=false
    echo "FLAGS_graph_metapath_split_opt is false"
fi
