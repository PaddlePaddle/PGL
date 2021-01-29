#!/bin/bash
set -x
echo "$(date +%Y%m%d_%H%M%S) $0 started"
mpirun $(cd $(dirname $0); pwd)/process_graph_data.sh > tmp.log  2>&1
echo "$(date +%Y%m%d_%H%M%S) $0 finished"
