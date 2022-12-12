#!/bin/bash
SOURCE_HOME=$(readlink -f $(dirname ${BASH_SOURCE[0]}) )/

LOG_DIR=${SOURCE_HOME}/logs
[ ! -d ${LOG_DIR} ] && mkdir -p ${LOG_DIR}

config_file=$1
cp ${config_file} ${SOURCE_HOME}/src/config.yaml

pushd ${SOURCE_HOME}/src
source ./env.sh
python -u cluster_train_and_infer.py 2>&1 | tee ${LOG_DIR}/run.log
popd 
