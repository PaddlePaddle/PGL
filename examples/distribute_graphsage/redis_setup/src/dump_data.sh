filter(){
    lines=`cat $1`
    rm $1
    for line in $lines; do
        remote_host=`echo $line | cut -d":" -f1`
        remote_port=`echo $line | cut -d":" -f2`
        nc -z $remote_host $remote_port
        if [[ $? == 0 ]]; then
            echo $line >> $1
        fi
    done
}

dump_data(){
    filter $server_list

    python ./src/start_cluster.py --server_list $server_list --replicas 0

    address=`head -n 1 $server_list`

    ip=`echo $address | cut -d":" -f1`
    port=`echo $address | cut -d":" -f2`

    python ./src/build_graph.py --startup_host $ip        \
        --startup_port $port        \
        --mode node_feat        \
        --node_feat_path $feat_fn       \
        --num_nodes $num_nodes

    # build edge index
    python ./src/build_graph.py --startup_host $ip \
        --startup_port $port \
        --mode edge_index \
        --edge_path $edge_path \
        --num_nodes $num_nodes

    # build edge id
    #python ./src/build_graph.py --startup_host $ip \
    #    --startup_port $port \
    #    --mode edge_id \
    #    --edge_path $edge_path \
    #    --num_nodes $num_nodes

    # build graph attr
    python ./src/build_graph.py --startup_host $ip \
        --startup_port $port \
        --mode graph_attr \
        --edge_path $edge_path \
        --num_nodes $num_nodes

}

if [ $# -ne 4 ]; then
    echo 'sh edge_path server_list num_nodes feat_fn'
    exit
fi
num_nodes=$3
server_list=$2
edge_path=$1
feat_fn=$4

dump_data

