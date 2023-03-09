#!/bin/bash
SOURCE_HOME=$(readlink -f $(dirname ${BASH_SOURCE[0]}) )/
PGLBOX_HOME=${SOURCE_HOME}/../

LOG_DIR=${PGLBOX_HOME}/logs
[ ! -d ${LOG_DIR} ] && mkdir -p ${LOG_DIR}

source ${PGLBOX_HOME}/scripts/util.sh
orig_config_file=$1
cp ${orig_config_file} ${PGLBOX_HOME}/src/config.yaml
config_file="${PGLBOX_HOME}/src/config.yaml"

fs_name=`parse_yaml2 ${config_file} graph_data_fs_name`
fs_ugi=`parse_yaml2 ${config_file} graph_data_fs_ugi`
export HADOOP_HOME=`parse_yaml2 ${config_file} hadoop_home`
export hadoop="$HADOOP_HOME/bin/hadoop fs -D fs.default.name=$fs_name -D hadoop.job.ugi=$fs_ugi"

train_mode=`parse_yaml2 ${config_file} train_mode`
if [ "${train_mode}" = "online_train" ]; then
    # online train
    graph_data_hdfs_path=`parse_yaml2 ${config_file} graph_data_hdfs_path`
    graph_data_local_path=`parse_yaml2 ${config_file} graph_data_local_path`
    time_delta=`parse_yaml2 $config_file time_delta`
    newest_time=`parse_yaml2 $config_file start_time`
    # 将最新日期地址写到配置文件中
    is_existed=`cat ${config_file} | grep newest_time | wc -l`
    if [ ${is_existed} -eq 0 ]; then
        echo "newest_time: ${newest_time}" >> ${config_file}
    else
        modify_yaml ${config_file} newest_time ${newest_time}
    fi

    while true; do
        # 检测是否有最新的图数据产生
        cur_graph_data_hdfs_path=${graph_data_hdfs_path}/${newest_time}
        if [[ ${cur_graph_data_hdfs_path} =~ "hdfs:" ]]; then
            # hadoop
            while true; do
                $hadoop -test -e ${cur_graph_data_hdfs_path}/to.hadoop.done
                if [ $? -eq 0 ]; then
                    break
                else
                    sleep 60
                    continue
                fi
            done
        else
            # local machine
            while true; do
                if [ -f ${cur_graph_data_hdfs_path}/to.hadoop.done ]; then
                    break
                else
                    sleep 60
                    continue
                fi
            done
        fi

        # 创建新的日志目录
        LAST_DATE=`echo ${newest_time} | awk -F"/" '{print $1}'`
        LAST_HOUR=`echo ${newest_time} | awk -F"/" '{print $2}'`
        CUR_LOG_DIR=${LOG_DIR}/${LAST_DATE}_${LAST_HOUR}
        [ ! -d ${CUR_LOG_DIR} ] && mkdir -p ${CUR_LOG_DIR}

        # 开始加载最新图数据
        sh -x ${SOURCE_HOME}/download_graph_data.sh ${cur_graph_data_hdfs_path} ${graph_data_local_path} > ${CUR_LOG_DIR}/graph_data.log 2>&1

        # train
        pushd ${PGLBOX_HOME}/src
        source ./env.sh
        python -u cluster_train_and_infer.py 2>&1 | tee ${CUR_LOG_DIR}/run.log
        popd

        # 更新到下一个日期
        LAST_DATE=`echo ${newest_time} | awk -F"/" '{print $1}'`
        LAST_HOUR=`echo ${newest_time} | awk -F"/" '{print $2}'`
        newest_time=`date -d "${LAST_DATE} ${LAST_HOUR} ${time_delta} hour" +"%Y%m%d/%H"`
        modify_yaml ${config_file} newest_time ${newest_time}
    done
else
    # normal train
    pushd ${PGLBOX_HOME}/src
    source ./env.sh
    python -u cluster_train_and_infer.py 2>&1 | tee ${LOG_DIR}/run.log
    popd
fi
