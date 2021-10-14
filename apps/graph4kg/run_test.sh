#!/bin/bash
#set -xe

LOG_NAME='tmp.log'

echo "" > $LOG_NAME

for i in {1..10}
do
    echo 'running '$i 
    python -m unittest mp_speed_test.py >> $LOG_NAME 2>&1
done

for name in "test_paddle" 'test_pickle' 'test_share_ndarray' 'test_pure' 'test_deepcopy'
do
    echo $name
    grep  $name $LOG_NAME | awk  '{print $4}' | awk '{x+=$0;y+=$0^2}END{printf "\t mean: %f std: %f\n", x/NR, sqrt((y-x^2/NR)/(NR-1)) }'
done
