#!/bin/bash

# 需要分片的数量，目前只能是1000
part_num=1000
# 原始图数据所在目录
base_input_dir="/your/path/to/graph_data"
# 分片后的数据保存目录
base_output_dir="/your/path/to/preprocessed_graph_data"

# 节点文件最后一级目录，一般node_type_list 只有一个目录，不同类型节点可以混合在一起。
node_type_list=(node_types)
# 边类型文件最后一级目录，根据自己的边类型的数据进行增减。
edge_type_list=(author2paper paper2paper author2inst)

for ntype in ${node_type_list[*]}; do
    # sharding node_types
    echo "processing ${ntype}"
    python graph_sharding.py --input_dir ${base_input_dir}/${ntype} \
                             --output_dir ${base_output_dir}/${ntype} \
                             --part_num ${part_num} \
                             --node_type_shard
done

for etype in ${edge_type_list[*]}; do
    # sharding edge_types
    echo "processing ${etype}"
    python graph_sharding.py --input_dir ${base_input_dir}/${etype} \
                             --output_dir ${base_output_dir}/${etype} \
                             --part_num ${part_num} \
                             --symmetry
done
