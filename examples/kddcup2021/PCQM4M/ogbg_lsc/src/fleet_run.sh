#!/bin/bash
if [ $# != 2 ]; then
    echo "gpus and task_name should be specified!! "
    echo "usage: sh fleet_run.sh "0,1,2,3" train.task.0420"
fi

if [ "$1" = "all" ]; then
    dev="0,1,2,3,4,5,6,7"
else
    dev=$1
fi

CUDA_VISIBLE_DEVICES=$dev fleetrun --log_dir ../fleet_logs/$2 main.py --config config.yaml --task_name $2
