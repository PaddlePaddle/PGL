#!/bin/bash
pushd src
source ./env.sh
python cluster_train_and_infer.py
popd 
