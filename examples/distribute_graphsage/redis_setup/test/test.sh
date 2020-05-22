#!/bin/bash 

source ./redis_graph.cfg

url=`head -n1 $server_list`
shuf $edge_path | head -n 1000 | python ./test/test_redis_graph.py $url

