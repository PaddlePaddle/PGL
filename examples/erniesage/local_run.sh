#!/bin/bash 

set -x
config=${1:-"./config.yaml"}
unset http_proxy https_proxy

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

transpiler_local_train(){
    export PADDLE_TRAINERS_NUM=1
    export PADDLE_PSERVERS_NUM=1
    export PADDLE_PORT=6206
    export PADDLE_PSERVERS="127.0.0.1"
    export BASE="./local_dir"
    echo `which python`
    if [ -d ${BASE} ]; then
        rm -rf ${BASE}
    fi 
    mkdir ${BASE}
    rm job_id
    for((i=0;i<${PADDLE_PSERVERS_NUM};i++))
    do
        echo "start ps server: ${i}"
        TRAINING_ROLE="PSERVER" PADDLE_TRAINER_ID=${i} sh job.sh local $config \
            &> $BASE/pserver.$i.log &
        echo $! >> job_id
    done
    sleep 3s 
    for((j=0;j<${PADDLE_TRAINERS_NUM};j++))
    do
        echo "start ps work: ${j}"
        TRAINING_ROLE="TRAINER" PADDLE_TRAINER_ID=${j} sh job.sh local $config \
        echo $! >> job_id
    done
}

collective_local_train(){
    export PATH=./python27-gcc482-gpu/bin/:$PATH
    echo `which python`
    python -m paddle.distributed.launch train.py --conf $config
    python -m paddle.distributed.launch infer.py --conf $config
}

eval $(parse_yaml $config)
unalias python

python3 ./preprocessing/dump_graph.py -i $input_data -o $graph_path --encoding $encoding \
    -l $max_seqlen --vocab_file $ernie_vocab_file

if [[ $learner_type == "cpu" ]];then
    transpiler_local_train
fi
if [[ $learner_type == "gpu" ]];then
    collective_local_train
fi
