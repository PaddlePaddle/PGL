#!/bin/bash

part_num=1000
base_input_dir="/your/path/to/graph_data"
base_output_dir="/your/path/to/preprocessed_graph_data"

# sharding node_types
python graph_sharding.py --input_dir ${base_input_dir}/node_types \
                         --output_dir ${base_output_dir}/node_types \
                         --part_num ${part_num} \
                         --node_type_shard

for etype in author2paper paper2paper author2inst; do
    python graph_sharding.py --input_dir ${base_input_dir}/${etype} \
                             --output_dir ${base_output_dir}/${etype} \
                             --part_num ${part_num} \
                             --symmetry
done
