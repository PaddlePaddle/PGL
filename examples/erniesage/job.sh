
unset http_proxy https_proxy
set -x
mode=${1:-local}
config=${2:-"./config.yaml"}

function parse_yaml {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]}}
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
         printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
      }
   }'
}
eval $(parse_yaml $config)

export CPU_NUM=$CPU_NUM
export FLAGS_rpc_deadline=3000000 
export FLAGS_rpc_retry_times=1000

if [[ $async_mode == "True" ]];then
    echo "async_mode is True"
else
    export FLAGS_communicator_send_queue_size=1
    export FLAGS_communicator_min_send_grad_num_before_recv=0
    export FLAGS_communicator_max_merge_var_num=1 # important! 
    export FLAGS_communicator_merge_sparse_grad=0
fi

export FLAGS_communicator_recv_wait_times=5000000

mkdir -p output

python ./train.py --conf $config
if [[ $TRAINING_ROLE == "TRAINER" ]];then
    python ./infer.py --conf $config
fi
