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

import sys
import socket
import argparse
import os
temp = """port %s
bind %s
daemonize yes
pidfile  /var/run/redis_%s.pid
cluster-enabled yes
cluster-config-file nodes.conf
cluster-node-timeout 50000
logfile "redis.log"
appendonly yes"""


def gen_config(ports):
    if len(ports) == 0:
        raise ValueError("No ports")
    ip = socket.gethostbyname(socket.gethostname())
    print("Generate redis conf")
    for port in ports:
        try:
            os.mkdir("%s" % port)
        except:
            print("port %s directory already exists" % port)
            pass
        with open("%s/redis.conf" % port, 'w') as f:
            f.write(temp % (port, ip, port))

    print("Generate Start Server Scripts")
    with open("start_server.sh", "w") as f:
        f.write("set -x\n")
        for ind, port in enumerate(ports):
            f.write("# %s %s start\n" % (ip, port))
            if ind > 0:
                f.write("cd ..\n")
            f.write("cd %s\n" % port)
            f.write("redis-server redis.conf\n")
            f.write("\n")

    print("Generate Stop Server Scripts")
    with open("stop_server.sh", "w") as f:
        f.write("set -x\n")
        for ind, port in enumerate(ports):
            f.write("# %s %s shutdown\n" % (ip, port))
            f.write("redis-cli -h %s -p %s shutdown\n" % (ip, port))
            f.write("\n")

    with open("server.list", "w") as f:
        for ind, port in enumerate(ports):
            f.write("%s:%s\n" % (ip, port))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gen_redis_conf')
    parser.add_argument('--ports', nargs='+', type=int, default=[])
    args = parser.parse_args()
    gen_config(args.ports)
