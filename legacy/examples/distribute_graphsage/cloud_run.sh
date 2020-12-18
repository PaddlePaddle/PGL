#!/bin/bash 
set -x
mode=${1}

source ./utils.sh
unset http_proxy https_proxy

source ./local_config
if [ ! -d ${log_dir} ]; then
    mkdir ${log_dir}
fi 

for((i=0;i<${PADDLE_PSERVERS_NUM};i++))
do
    echo "start ps server: ${i}"
    echo $log_dir
    TRAINING_ROLE="PSERVER" PADDLE_TRAINER_ID=${i} sh job.sh &> $log_dir/pserver.$i.log & 
done
sleep 10s 
for((j=0;j<${PADDLE_TRAINERS_NUM};j++))
do
    echo "start ps work: ${j}"
    TRAINING_ROLE="TRAINER" PADDLE_TRAINER_ID=${j} sh job.sh &> $log_dir/worker.$j.log &
done
tail -f $log_dir/worker.0.log
