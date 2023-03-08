#!/bin/bash
SOURCE_HOME=$(readlink -f $(dirname ${BASH_SOURCE[0]}) )/
PGLBOX_HOME=${SOURCE_HOME}/../

source ${SOURCE_HOME}/util.sh
config_file="${PGLBOX_HOME}/src/config.yaml"
date && echo 'begin download graph data...'

function download_data_by_hadoop() {
    pushd ${SOURCE_HOME}/../src
    local graph_data_hdfs_path=$1
    local graph_data_local_path=$2
    graph_data_fs_name=`parse_yaml2 ${config_file} graph_data_fs_name`
    graph_data_fs_ugi=`parse_yaml2 ${config_file} graph_data_fs_ugi`
    rm -rf ${graph_data_local_path}
    mkdir -p ${graph_data_local_path}
    python ${SOURCE_HOME}/download_graph_data.py ${graph_data_hdfs_path} \
                                                 ${graph_data_fs_name} \
                                                 ${graph_data_fs_ugi} \
                                                 ${HADOOP_HOME} \
                                                 ${graph_data_local_path}
    popd
}

function copy_data_from_local_machine() {
    pushd ${SOURCE_HOME}/../src
    local graph_data_hdfs_path=$1
    local graph_data_local_path=$2
    rm -rf ${graph_data_local_path}
    mkdir -p ${graph_data_local_path}
    cp -r ${graph_data_hdfs_path}/* ${graph_data_local_path} 
    popd
}


echo "======================== [BUILD_INFO] download_gdata =============================="
hdfs_path=$1
local_path=$2
data_flag=$3
echo "download or copy data from [${hdfs_path}] to [${local_path}]"
if [ ${data_flag} -eq 0 ]; then
    copy_data_from_local_machine ${hdfs_path} ${local_path}
elif [ ${data_flag} -eq 1 ]; then
    download_data_by_hadoop ${hdfs_path} ${local_path}
fi
[ $? -ne 0 ] && fatal_error "download data failed"
