function parse_yaml {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]}}
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
         printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
      }
   }'
}

# 解释yaml配置选项并转化为环境变量
smoothing_config=`readlink -f $1`
eval $(parse_yaml ${smoothing_config})
set -xou pipefail

source ./hadoop_func.sh

TASK_NAME=$smoothing_task_name
edges_path="${hadoop_edges_path}/*"

# preprocess_node ${hadoop_graph_nodes} ${hadoop_output_path}/pnf
func0(){
if $indegree; then
    get_indegree ${edges_path} ${hadoop_embed_path} ${hadoop_output_path}/0_hop_ppnp
fi
}

func1(){
for r in `seq 1 ${khop}`; do
    last_hop=`echo $r - 1 | bc`
    process_ppnp ${edges_path} ${hadoop_output_path}/${last_hop}_hop_ppnp ${hadoop_output_path}/${r}_hop_ppnp ${r}
done
}

func0
func1
