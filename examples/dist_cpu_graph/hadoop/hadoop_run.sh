set -x

SOURCE_HOME=$(readlink -f $(dirname ${BASH_SOURCE[0]}) )/../src/
pushd $SOURCE_HOME
source ./util.sh
popd

# config
task_name=graph_data_sharding

map_task=200
reduce_task=${shard_num}

if [ "${symmetry}" = "True" ] || [ "${symmetry}" = "true" ]; then
    bidirection="yes"
else
    bidirection="no"
fi

hadoop_bin='hadoop'

# hadoop function
function clear_hadoop_files() {
   local file_path=$1
   ${hadoop_bin} fs -test -e ${file_path}
   if [ $? -eq 0 ]; then
       ${hadoop_bin} fs -rmr ${file_path}
   fi
}

function graph_data_sharding() {
    local hadoop_input_path=$1
    local hadoop_output_path=$2
    local symmetry=$3
    local node_type_shard=$4
    clear_hadoop_files ${hadoop_output_path}
    
    $hadoop_bin streaming \
        -D mapred.job.name=${task_name} \
        -D mapred.job.map.capacity=10000 \
        -D mapred.job.reduce.capacity=5000 \
        -D mapred.map.tasks=${map_task} \
        -D mapred.reduce.tasks=${reduce_task} \
        -D stream.memory.limit=2000 \
        -D mapred.map.over.capacity.allowed=true \
        -D mapred.job.priority=HIGH \
        -D abaci.split.optimize.enable=true \
        -D abaci.job.base.environment=default \
        -D mapred.output.key.comparator.class="org.apache.hadoop.mapred.lib.KeyFieldBasedComparator" \
        -D mapred.text.key.comparator.options="-k1,1" \
        -input  ${hadoop_input_path} \
        -output ${hadoop_output_path} \
        -mapper "export LANG=en_US.UTF-8;export LC_ALL=en_US.UTF-8;export LC_CTYPE=en_US.UTF-8; python ./hashcode.py map ${reduce_task} ${symmetry} ${node_type_shard}" \
        -reducer "export LANG=en_US.UTF-8;export LC_ALL=en_US.UTF-8;export LC_CTYPE=en_US.UTF-8; python ./hashcode.py reduce" \
        -file hashcode.py \

}

hadoop_base_output_path="xxx"

# data config
# NOTE: hadoop_output_path 的最后一级目录名称需要跟../src/config.yaml 文件中的edges_files参数的edge_file_or_directory保持一致
# 例如: 第一个hadoop_output_path 最后一级目录是t2f, 那么在../src/config.yaml 中edges_files为: t2f:t2f 
# 其它的以此类推
# 然后, ../src/config.yaml 中的graph_data_hdfs_path 设置为 
hadoop_input_path="xxx/t2f_edges.txt"
hadoop_output_path="${hadoop_base_output_path}/t2f"
graph_data_sharding ${hadoop_input_path} ${hadoop_output_path} ${bidirection} "no"

hadoop_input_path="xxx/u2f_edges.txt"
hadoop_output_path="${hadoop_base_output_path}/u2f"
graph_data_sharding ${hadoop_input_path} ${hadoop_output_path} ${bidirection} "no"

hadoop_input_path="xxx/u2t_edges.txt"
hadoop_output_path="${hadoop_base_output_path}/u2t"
graph_data_sharding ${hadoop_input_path} ${hadoop_output_path} ${bidirection} "no"

hadoop_node_type_path="xxx/node_types.txt"
#NOTE:  hadoop_node_type_output_path 的最后一级目录要与../src/config.yaml中的node_types_file保持一致
hadoop_node_type_output_path="${hadoop_base_output_path}/node_types"
graph_data_sharding ${hadoop_node_type_path} ${hadoop_node_type_output_path} "no" "yes"

