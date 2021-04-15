# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import threading
import multiprocessing
import time

from paddle.distributed.fleet.base.private_helper_function import wait_server_ready

from pgl.utils.logger import log
from pgl.distributed.dist_graph import DistGraphServer, DistGraphClient
from pgl.distributed import helper


class Dist(object):
    """dist config"""

    def __init__(self, MPI):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.nrank = self.comm.Get_size()

    def worker_index(self):
        """get worker index"""
        return self.rank

    def worker_num(self):
        """get worker number"""
        return self.nrank

    def allgather(self, obj):
        """allgather infomation"""
        recv_obj = self.comm.allgather(obj)
        return recv_obj

    def barrier_worker(self):
        """barrier between worker"""
        self.comm.Barrier()

    def get_local_ip(self):
        """get local ip"""
        import socket
        self._ip = socket.gethostbyname(socket.gethostname())
        return self._ip


def start_client(config, ip_addr, server_num, server_id, shard_num=1000):
    wait_server_ready(ip_addr)
    graph_client = DistGraphClient(
        config, shard_num=shard_num, ip_config=ip_addr, client_id=server_id)

    if server_id == 0:
        graph_client.load_edges()
        graph_client.load_node_types()


def launch_graph_service(config,
                         ip_config,
                         server_id,
                         mode,
                         shard_num=1000,
                         port=8245):
    """

    Args:
        config: a yaml configure file for distributed graph service

        ip_config: list of IP address or a path of IP configuration file

            For example, the following TXT shows a 4-machine configuration:

                172.31.50.123:8245
                172.31.50.124:8245
                172.31.50.125:8245
                172.31.50.126:8245

        server_id: The graph server rank

        shard_num: The total sharding number for the graph. 
            Each server is responsible for part of shards.

        port: int, port number, default is 8245

    Return:
        A DistGraphClient for handling GraphServer 

    """
    if ip_config is not None:
        ip_addr = helper.load_ip_addr(ip_config)
        ip_addr = ip_addr.split(";")

    if mode == "mpi":
        try:
            from mpi4py import MPI
            fleet = Dist(MPI)
        except Exception as e:
            log.info(e)
            raise

        server_id = fleet.worker_index()

        if ip_config is None:
            local_ip_addr = ":".join([fleet.get_local_ip(), str(port)])
            ip_addr = fleet.allgather(local_ip_addr)

    server_num = len(ip_addr)

    if server_id == 0:
        args = (config, ip_addr, server_num, server_id, shard_num)
        process = multiprocessing.Process(target=start_client, args=args)
        process.start()

    graph_server = DistGraphServer(
        config, shard_num, ip_addr, server_id=server_id, is_block=False)
    while True:
        time.sleep(10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Launch Distributed Graph Server')
    parser.add_argument("--mode", type=str, default="local", help="mpi mode")
    parser.add_argument(
        "--ip_config",
        type=str,
        default=None,
        help="list of IP address or a path of IP configuration file")
    parser.add_argument(
        "--config", type=str, help="Distributed Graph Server Config (yaml)")
    parser.add_argument(
        "--server_id", type=int, default=0, help="The Rank of the server.")
    parser.add_argument(
        "--shard_num", type=int, default=1000, help="Sharding for graph")
    parser.add_argument("--port", type=int, default=8245, help="port number")
    args = parser.parse_args()
    log.info(args)

    launch_graph_service(args.config, args.ip_config, args.server_id,
                         args.mode, args.shard_num, args.port)
