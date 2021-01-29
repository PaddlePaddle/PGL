#!/bin/bash
set -x
echo "$(date +%Y%m%d_%H%M%S) $0 started"
SOURCE_HOME=$(readlink -f $(dirname ${BASH_SOURCE[0]}) )/
pushd $SOURCE_HOME

sh download_graph_data.sh 2>&1 | tee graph_data_${OMPI_COMM_WORLD_RANK}.log

log "finished"
popd
