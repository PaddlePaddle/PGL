#!/bin/bash
cd "$(dirname "$0")"

cross="cross1"
cp config.yaml run_cross.yaml

task_name=`cat run_cross.yaml | head -n 1 | cut -d' ' -f2`

new_task_name="${task_name}_${cross}"
sed -i "s|^task_name: .*$|task_name: $new_task_name|" run_cross.yaml
sed -i "s|^split_mode: .*$|split_mode: $cross|" run_cross.yaml

export CUDA_VISIBLE_DEVICES=$1
python main.py --config run_cross.yaml --task_name $new_task_name &

sleep 60

cross="cross2"
cp config.yaml run_cross.yaml

new_task_name="${task_name}_${cross}"
sed -i "s|^task_name: .*$|task_name: $new_task_name|" run_cross.yaml
sed -i "s|^split_mode: .*$|split_mode: $cross|" run_cross.yaml

export CUDA_VISIBLE_DEVICES=$2
python main.py --config run_cross.yaml --task_name $new_task_name &

rm run_cross.yaml
