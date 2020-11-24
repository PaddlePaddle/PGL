#!/bin/bash 

config=${1:-"./config/erniesage_link_predict.yaml"}

python ./preprocessing/dump_graph.py --conf $config

unset http_proxy https_proxy
python -m paddle.distributed.launch link_predict.py --conf $config
python -m paddle.distributed.launch link_predict.py --conf $config --do_predict
