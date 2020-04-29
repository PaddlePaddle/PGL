#!/bin/bash 

set -x
source ./pgl_deepwalk.cfg
source ./local_config

unset http_proxy https_proxy

# build train_data
trainer_num=`echo $PADDLE_PORT | awk -F',' '{print NF}'`
rm -rf train_data && mkdir -p train_data 
cd train_data
if [[ $build_train_data == True ]];then
    seq 0 $((num_nodes-1)) | shuf | split -l $((num_nodes/trainer_num/CPU_NUM+1))
else
    for i in `seq 1 $trainer_num`;do
        touch $i
    done
fi
cd - 

# mkdir workspace
if [ -d ${BASE} ]; then
    rm -rf ${BASE}
fi 
mkdir ${BASE}

# start ps
for((i=0;i<${PADDLE_PSERVERS_NUM};i++))
do
    echo "start ps server: ${i}"
    echo $BASE
    TRAINING_ROLE="PSERVER" PADDLE_TRAINER_ID=${i} sh job.sh &> $BASE/pserver.$i.log & 
done
sleep 5s 

# start trainers
for((j=0;j<${PADDLE_TRAINERS_NUM};j++))
do
    echo "start ps work: ${j}"
    TRAINING_ROLE="TRAINER" PADDLE_TRAINER_ID=${j} sh job.sh &> $BASE/worker.$j.log &
done
