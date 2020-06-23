# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse


def build_clusters(server_list, replicas):
    servers = []
    with open(server_list) as f:
        for line in f:
            servers.append(line.strip())
    cmd = "echo yes | redis-cli --cluster create"
    for server in servers:
        cmd += ' %s ' % server
    cmd += '--cluster-replicas %s' % replicas
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='start_cluster')
    parser.add_argument('--server_list', type=str, required=True)
    parser.add_argument('--replicas', type=int, default=0)
    args = parser.parse_args()
    build_clusters(args.server_list, args.replicas)
