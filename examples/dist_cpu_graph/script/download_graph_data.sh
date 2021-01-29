#!/bin/bash
log() { echo "$(date +%Y%m%d_%H%M%S) $0 $@" ; }
set -x
log "started"
env
SOURCE_HOME=$(readlink -f $(dirname ${BASH_SOURCE[0]}) )/../src/
pushd $SOURCE_HOME
source ./util.sh
export PATH=/opt/compiler/gcc-4.8.2/bin:$PATH
export PYTHON_HOME=`readlink -f ./python_gcc482`
export PATH=$PYTHON_HOME/bin:$PATH
export LDFLAGS="-L$PYTHON_HOME/lib"
export TRAINING_ROLE="TRAINER"
export LD_LIBRARY_PATH=`readlink -f ./`:$LD_LIBRARY_PATH
which python

# can not alias, do not know why
# alias hfs="hadoop fs -Dfs.default.name=$fs_name  -Dhadoop.job.ugi=$fs_ugi -Ddfs.client.block.write.retries=15 -Ddfs.rpc.timeout=300000 -Ddfs.delete.trash=1"
# hfs -ls $graph_data

rm -rf ./graph_data
shard=`echo $shard | xargs echo -n`  # remove space
if [ "$shard" = "True" ] || [ "$shard" = "true" ]; then
    mkdir -p ./graph_data
    # worker_index=`printf "%05d" ${OMPI_COMM_WORLD_RANK}`
    worker_index=${OMPI_COMM_WORLD_RANK}
    shard_num=$((shard_num-1))  # for seq 0, shard_num - 1 

    IFS=',' read -r -a array <<< "$edge_files"  # [etype1:e_files1, etype2:e_files2]
    for element in "${array[@]}"
    do
        # echo "$element"
        IFS=':' read -r -a fields <<< "$element"
        echo "${fields[0]}"
        echo "${fields[1]}"
        mkdir -p ./graph_data/${fields[1]}
        for part_num in `seq 0 ${shard_num}`; do
            residue=$((${part_num} % ${mpi_trainer_num}))  # which part can be pulled in this machine
            if [ $residue = $worker_index ]; then
                num=`printf "%05d" ${part_num}`
                hadoop fs -Dfs.default.name=$fs_name -Dhadoop.job.ugi=$fs_ugi -get $graph_data_hdfs_path/${fields[1]}/part-${num} ./graph_data/${fields[1]}
            fi
        done
    done

    # for node_types file
    mkdir -p ./graph_data/${node_types_file}
    for part_num in `seq 0 ${shard_num}`; do
        residue=$((${part_num} % ${mpi_trainer_num}))
        if [ $residue = $worker_index ]; then
            num=`printf "%05d" ${part_num}`
            hadoop fs -Dfs.default.name=$fs_name -Dhadoop.job.ugi=$fs_ugi -get $graph_data_hdfs_path/${node_types_file}/part-${num} ./graph_data/${node_types_file}
        fi
    done

else
    hadoop fs -Dfs.default.name=$fs_name -Dhadoop.job.ugi=$fs_ugi -get $graph_data_hdfs_path ./graph_data
fi

python -m pgl.distributed.graph_service 2>&1 | tee graph_server.log

log "finished"
popd
