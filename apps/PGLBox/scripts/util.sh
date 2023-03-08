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

function parse_yaml2 {
    local file=$1
    local key=$2
    grep ^$key $file | sed s/#.*//g | grep $key | awk -F':' -vOFS=':' '{$1=""; print $0;}' | awk '{print $2;}' | sed 's/ //g; s/"//g'
}

function modify_yaml {
    local file=$1
    local key=$2
    local value=$3
    sed -i "s|^${key}: .*$|${key}: ${value}|" ${file}
}

function fatal_error() {
    d=`date`
    echo -e "$d: FATAL: " "$1" >> /dev/stderr
    exit 1
}
