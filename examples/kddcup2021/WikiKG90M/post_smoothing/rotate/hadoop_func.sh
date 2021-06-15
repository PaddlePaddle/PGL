function clear_hadoop_files() {
   local file_path=$1
   ${local_hadoop_bin} fs -test -e ${file_path}
   if [ $? -eq 0 ]; then
       ${local_hadoop_bin} fs -rmr ${file_path}
   fi
}

function get_indegree() {
    local hadoop_input_edges_path=$1
    local hadoop_last_nfeat_path=$2
    local hadoop_output=$3
    clear_hadoop_files ${hadoop_output}
    
    $local_hadoop_bin streaming \
        -D mapred.job.name=$TASK_NAME \
        -D mapred.job.map.capacity=10000 \
        -D mapred.job.reduce.capacity=5000 \
        -D mapred.map.tasks=${map_task} \
        -D mapred.reduce.tasks=${reduce_task} \
        -D stream.memory.limit=10000 \
        -D mapred.map.over.capacity.allowed=true \
        -D mapred.job.priority=HIGH \
        -D abaci.split.optimize.enable=true \
        -D abaci.job.base.environment=default \
        -D mapred.output.key.comparator.class="org.apache.hadoop.mapred.lib.KeyFieldBasedComparator" \
        -D mapred.text.key.comparator.options="-k1,1" \
        -input  ${hadoop_input_edges_path} \
        -input  ${hadoop_last_nfeat_path} \
        -output ${hadoop_output} \
        -mapper "export LANG=en_US.UTF-8;export LC_ALL=en_US.UTF-8;export LC_CTYPE=en_US.UTF-8; ./python37/bin/python ./hadoop_prep.py -m indegree_map" \
        -reducer "export LANG=en_US.UTF-8;export LC_ALL=en_US.UTF-8;export LC_CTYPE=en_US.UTF-8; sh sort.sh | ./python37/bin/python ./hadoop_prep.py -m indegree_reduce" \
        -file hadoop_prep.py \
        -file smoothing_config.yaml \
        -file sort.sh \
        -cacheArchive ${python_package}#python37 \

}

function process_ppnp() {
    local hadoop_input_edges_path=$1
    local hadoop_indegree_nfeat_path=$2
    local hadoop_output=$3
    local runs=$4
    clear_hadoop_files ${hadoop_output}_tmp
    clear_hadoop_files ${hadoop_output}
    
    $local_hadoop_bin streaming \
        -D mapred.job.name=$TASK_NAME \
        -D mapred.job.map.capacity=10000 \
        -D mapred.job.reduce.capacity=5000 \
        -D mapred.map.tasks=${map_task} \
        -D mapred.reduce.tasks=${reduce_task} \
        -D stream.memory.limit=10000 \
        -D mapred.map.over.capacity.allowed=true \
        -D mapred.job.priority=VERY_HIGH \
        -D abaci.split.optimize.enable=true \
        -D abaci.job.base.environment=default \
        -D mapred.output.key.comparator.class="org.apache.hadoop.mapred.lib.KeyFieldBasedComparator" \
        -D mapred.text.key.comparator.options="-k1,1 -k2,2n" \
        -input  ${hadoop_input_edges_path} \
        -input  ${hadoop_indegree_nfeat_path} \
        -output ${hadoop_output}_tmp \
        -mapper "export LANG=en_US.UTF-8;export LC_ALL=en_US.UTF-8;export LC_CTYPE=en_US.UTF-8; ./python37/bin/python ./hadoop_prep.py -m merge -r ${runs}" \
        -reducer "export LANG=en_US.UTF-8;export LC_ALL=en_US.UTF-8;export LC_CTYPE=en_US.UTF-8; sh sort.sh | ./python37/bin/python ./hadoop_prep.py -m join_src" \
        -file hadoop_prep.py \
        -file smoothing_config.yaml \
        -file sort.sh \
        -cacheArchive ${python_package}#python37 \

    $local_hadoop_bin streaming \
        -D mapred.job.name=$TASK_NAME \
        -D mapred.job.map.capacity=10000 \
        -D mapred.job.reduce.capacity=5000 \
        -D mapred.map.tasks=${map_task} \
        -D mapred.reduce.tasks=${reduce_task} \
        -D stream.memory.limit=10000 \
        -D mapred.map.over.capacity.allowed=true \
        -D mapred.job.priority=VERY_HIGH \
        -D abaci.split.optimize.enable=true \
        -D abaci.job.base.environment=default \
        -D mapred.output.key.comparator.class="org.apache.hadoop.mapred.lib.KeyFieldBasedComparator" \
        -D mapred.text.key.comparator.options="-k1,1" \
        -input  ${hadoop_output}_tmp \
        -output ${hadoop_output} \
        -mapper "cat" \
        -reducer "export LANG=en_US.UTF-8;export LC_ALL=en_US.UTF-8;export LC_CTYPE=en_US.UTF-8;  ./python37/bin/python ./hadoop_prep.py -m reduce_neigh -r ${runs}" \
        -file hadoop_prep.py \
        -file smoothing_config.yaml \
        -file ./wikikg90m_TransE_l2_relation.npy \
        -file sort.sh \
        -cacheArchive ${python_package}#python37 \

    clear_hadoop_files ${hadoop_output}_tmp

}

function ensemble_nfeat() {
    local hadoop_input_edges1_path=$1
    local hadoop_input_edges2_path=$2
    local hadoop_input_edges3_path=$3
    local hadoop_output=$4
    clear_hadoop_files ${hadoop_output}
    
    $local_hadoop_bin streaming \
        -D mapred.job.name=$TASK_NAME \
        -D mapred.job.map.capacity=10000 \
        -D mapred.job.reduce.capacity=5000 \
        -D mapred.map.tasks=${map_task} \
        -D mapred.reduce.tasks=${reduce_task} \
        -D stream.memory.limit=10000 \
        -D mapred.map.over.capacity.allowed=true \
        -D mapred.job.priority=VERY_HIGH \
        -D abaci.split.optimize.enable=true \
        -D abaci.job.base.environment=default \
        -D mapred.output.key.comparator.class="org.apache.hadoop.mapred.lib.KeyFieldBasedComparator" \
        -D mapred.text.key.comparator.options="-k1,1" \
        -input  ${hadoop_input_edges1_path} \
        -input  ${hadoop_input_edges2_path} \
        -input  ${hadoop_input_edges3_path} \
        -output ${hadoop_output} \
        -mapper "cat" \
        -reducer "export LANG=en_US.UTF-8;export LC_ALL=en_US.UTF-8;export LC_CTYPE=en_US.UTF-8; ./python37/bin/python ./hadoop_prep.py -m ensemble" \
        -file hadoop_prep.py \
        -file smoothing_config.yaml \
        -cacheArchive ${python_package}#python37 \

}
