#!/bin/bash
set -x

srcdir=./src

# Data preprocessing
python ./src/preprocess.py

# Download and compile redis
export PATH=$PWD/redis-5.0.5/src:$PATH
if [ ! -f ./redis.tar.gz ]; then
    curl https://codeload.github.com/antirez/redis/tar.gz/5.0.5 -o ./redis.tar.gz
fi
tar -xzf ./redis.tar.gz
cd ./redis-5.0.5/
make
cd -

# Install python deps
python -m pip install -U pip
pip install -r ./src/requirements.txt -U

# Run redis server
sh ./src/run_server.sh

# Dumping data into redis
source ./redis_graph.cfg
sh ./src/dump_data.sh $edge_path $server_list $num_nodes $node_feat_path

exit 0

