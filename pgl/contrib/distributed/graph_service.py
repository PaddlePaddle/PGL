# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""distributed
"""
from concurrent import futures
import pgl
import multiprocessing

import grpc
import sys

from pgl.contrib.distributed import graph_service_pb2
from pgl.contrib.distributed import graph_service_pb2_grpc

from tqdm import tqdm
from collections import defaultdict
import time
import os
import glob
from pgl.utils.logger import log
import contextlib
import socket
import numpy as np
from mpi4py import MPI
import time
import argparse
from contextlib import closing
import platform

import yaml

class Config:
    """ Global Configuration Management
    """
    def __init__(self, config_path):
        with open(config_path) as f:
            if hasattr(yaml, 'FullLoader'):
                self.config = yaml.load(f, Loader=yaml.FullLoader)
            else:
                self.config = yaml.load(f)
    
    def __getattr__(self, attr):
        return self.config[attr]

def versiontuple(v):
    return tuple(map(int, (v.split(".")[:2])))

def can_reuse_port():
    version = versiontuple(platform.release())
    base = versiontuple("3.9")
    if version >= base:
        log.info("Kernel Version is %s >= 3.9. Set reuse port = True" % platform.release())
        return True
    else:
        log.info("Kernel Version is %s < 3.9. Set reuse port = False" % platform.release())
        return False


SampleNeighborsReply = graph_service_pb2.SampleNeighborsReply
SampleNeighborsRequest = graph_service_pb2.SampleNeighborsRequest
NodeList = graph_service_pb2.NodeList

def find_free_port(ip):
    """find free port"""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind((ip, 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

class Dist(object):
    """dist config"""
    def __init__(self):
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

fleet = Dist()


class DistCPUGraphServer(object):
    """Distributed CPU Graph Server""" 
    def __init__(self, config, replicate_rate=0.0):
        self.config = config
        self.reuse_port = can_reuse_port()
        log.info("Graph Service %s" % fleet.worker_index())
        self.node_feat = {} 
        self.node_types = defaultdict(lambda : "")
        self.type_nodes = defaultdict(lambda : [])
        self.all_nodes = []

        self.map_node = defaultdict(lambda : {})
        self.replicate_part = int(fleet.worker_num() * replicate_rate) 
        self.graph = {}

    def is_replicate(self, node):
        """replicate nodes"""
        # No replicate
        return False

    def select_machine(self, nid):
        if self.config.shard:
            return nid % self.config.shard_num % fleet.worker_num()
        else:
            return nid % fleet.worker_num()

    def add_nodes(self, node_id, node_type):
        """ add nodes to server"""
        if node_id not in self.map_node[node_type]:
            self.map_node[node_type][node_id] = len(self.map_node[node_type])
        return self.map_node[node_type][node_id]

    def get_files(self, edge_file_or_dir):
        if os.path.isdir(edge_file_or_dir):
            files = glob.glob(os.path.join(edge_file_or_dir, "*"))
        else:
            files = [edge_file_or_dir]
        return files

    def load_graph_from_files(self, edge_file_or_dir, edge_type, symmetry):
        """ load graph from files """
        edges = []
        reverse_edges = []
        first_node_type, second_node_type = edge_type.split("2")

        cc = 0
        for file_ in self.get_files(edge_file_or_dir):
            with open(file_, "r") as f:
                for line in f:
                    e = line.strip().split()
                    src = np.uint64(e[0]).astype(np.int64)
                    dst = np.uint64(e[1]).astype(np.int64)

                    cc += 1
                    if cc % 1000000 == 0:
                        log.info("Load Edges EdgeType: %s Finished lines: %s" % (edge_type, cc))

                    if self.select_machine(src) == fleet.worker_index():
                        edges.append((dst, self.add_nodes(src, first_node_type + '2' + second_node_type) ))

     

                    if symmetry:
                        src, dst = dst, src
                        if self.select_machine(src) == fleet.worker_index() or self.is_replicate(src):
                            if (first_node_type != second_node_type):
                                reverse_edges.append((dst, self.add_nodes(src, second_node_type + '2' + first_node_type )))
                            else:
                                edges.append((dst, self.add_nodes(src, second_node_type + '2' + first_node_type )))
            
        self.graph[first_node_type + '2' + second_node_type] = pgl.graph.Graph(num_nodes=len(self.map_node[first_node_type + '2' + second_node_type]), edges=edges)
        self.graph[first_node_type + '2' + second_node_type].adj_dst_index
        self.graph[first_node_type + '2' + second_node_type] = self.to_mmap(first_node_type + '2' + second_node_type, self.graph[first_node_type + '2' + second_node_type])

        if symmetry and (first_node_type != second_node_type):
            self.graph[second_node_type + '2' + first_node_type] = pgl.graph.Graph(num_nodes=len(self.map_node[second_node_type + '2' + first_node_type]), edges=reverse_edges)
            self.graph[second_node_type + '2' + first_node_type].adj_dst_index
            self.graph[second_node_type + '2' + first_node_type] = self.to_mmap(second_node_type + '2' + first_node_type, self.graph[second_node_type + '2' + first_node_type]) 

        log.info("total %s edges have been loaded in %s" % (len(edges), '2'.join([first_node_type, second_node_type])))
        log.info("total %s reverse edges have been loaded in %s" % (len(reverse_edges), '2'.join([second_node_type, first_node_type])))

    def to_mmap(self, name, graph):
        path = os.path.join("mmap_graph_%s" % fleet.worker_index(), name)
        return graph.to_mmap(path)
        #graph.dump(path)
        #return pgl.graph.MemmapGraph(path)

    def load_node_feat(self, node_types_file_or_dir):
        """ load node feature"""
        cc = 0
        for file_ in self.get_files(node_types_file_or_dir):
            with open(file_, "r") as f:
                for line in f:

                    cc += 1
                    if cc % 1000000 == 0:
                        log.info("Load Node Features Finished lines: %s" % (cc, ))

                    try:
                        items = line.strip().split('\t')
                        node_id = np.uint64(items[1]).astype(np.int64)
                        if self.select_machine(node_id) != fleet.worker_index():
                            continue

                        n_type = items[0]
                        self.node_types[node_id] = n_type
                        self.type_nodes[n_type].append(node_id)
                        self.all_nodes.append(node_id)
                        self.node_feat[node_id] = ','.join(items[2:])
                    except:
                        continue
        self.all_nodes = np.array(self.all_nodes, dtype="int64")
        log.info("total %s nodes has been loaded in %s machine" \
                % (len(self.all_nodes), fleet.worker_index()))

    def start_service(self, max_workers=12):
        """ start service """
        fleet.barrier_worker()
        workers = []
       
        #port = "%s" % find_free_port(fleet.get_local_ip()) 
        if self.reuse_port:
            port = 8035 + fleet.worker_index()
            server_addresses = [fleet.get_local_ip() + ":%s" % port] * max_workers
        else:
            port = 8035 + fleet.worker_index() * max_workers
            server_addresses = [ fleet.get_local_ip() + ":" + str(p) for p in range(port, port+ max_workers)]

        all_endpoints = fleet.allgather(",".join(list(set(server_addresses))))

        with open("./server_endpoints", "w") as f:
            f.write("\n".join(all_endpoints) + '\n')
        log.info("Graph service address %s" % list(set(server_addresses)))

        for server_address in server_addresses:
            worker = multiprocessing.Process(target=self.start_service_worker, args=(server_address, ))
            worker.start()
            workers.append(worker)
       
        for worker in workers:
            worker.join()

    def start_service_worker(self, server_address):
        """ start service worker """
        if self.reuse_port:
            options = (('grpc.so_reuseport', 1),)
            server = grpc.server(futures.ThreadPoolExecutor(max_workers=12), options=options)
        else:
            server = grpc.server(futures.ThreadPoolExecutor(max_workers=12))

        graph_service_pb2_grpc.add_GraphServiceServicer_to_server(self, server)
        server.add_insecure_port(server_address)
        server.start()
        server.wait_for_termination()

    def SampleNeighbors(self, request, context):
        """ sample neighbors """
        max_size = request.max_size
        edge_type = request.edge_type

        ret_nodeid = []
        query_node = []
        query_node_id = []
        ret_neigh = []
        for n, node in enumerate(request.sample_nodes.nodeid):
            ret_nodeid.append(node)
            ret_neigh.append([])
            if node in self.map_node[edge_type]:
                query_node.append(self.map_node[edge_type][node])
                query_node_id.append(n)

        if len(query_node) > 0:
            neighs = self.graph[edge_type].sample_predecessor(query_node, max_degree=max_size)

            for n, neigh in zip(query_node_id, neighs):
                ret_neigh[n] = neigh

        ret_neigh = [ NodeList(nodeid=neigh) for neigh in ret_neigh]
        return SampleNeighborsReply(nodeid=NodeList(nodeid=ret_nodeid), neighbors=ret_neigh)

    def GetNodeFeat(self, request, context):
        """ get node feature"""
        feat_str = [] 
        for node in request.nodes.nodeid:
            if node in self.node_feat:
                feat_str.append(self.node_feat[node])
            else:
                feat_str.append("")
        return graph_service_pb2.GetNodeFeatReply(nodes=request.nodes, value="\n".join(feat_str))

    def NodeBatchIter(self, request, context):
        """ sampling nodes """
        shuffle = request.shuffle
        batch_size = request.batch_size
        node_type = request.node_type
        nrank = request.nrank
        rank = request.rank

        num_nodes = len(self.type_nodes[node_type])
        perm = np.arange(rank, num_nodes, step=nrank)
        if shuffle:
            np.random.shuffle(perm)
        batch_no = 0
        while batch_no < len(perm):
            batch = [ self.type_nodes[node_type][n] for n in perm[batch_no:batch_no + batch_size]]
            yield graph_service_pb2.SampleNodesReply(sample_nodes=NodeList(nodeid=batch))
            batch_no += batch_size

    def NegSampleNodes(self, request, context):
        """ sampling nodes """
        neg_sample_type = request.neg_sample_type
        neg_num = request.neg_num
        pos = [ n for n in request.nodeid.nodeid]
        if neg_sample_type == "m2v_plus":
            negs = []
            for s in pos:
                node_type = self.node_types[s]
                if node_type == "":
                    perm = np.random.randint(0, len(self.all_nodes), size=neg_num)
                    neg = self.all_nodes[perm]
                else:
                    num_nodes = len(self.type_nodes[node_type])
                    perm = np.random.randint(0, num_nodes, size=neg_num)
                    neg = [ self.type_nodes[node_type][n] for n in perm ]

                negs.append(neg)
            negs = np.vstack(negs).reshape(-1)
        else:
            perm = np.random.randint(0, len(self.all_nodes), size=len(pos) * neg_num)
            negs = self.all_nodes[perm]
        return graph_service_pb2.NegSampleNodesReply(sample_nodes=NodeList(nodeid=negs))

    def NodeTypes(self, request, context):
        return graph_service_pb2.NodeTypesReply(node_types=list(self.type_nodes.keys()))



if __name__ == "__main__":
    config = Config('./config.yaml')
    graph = DistCPUGraphServer(config)
    edge_path = config.edge_path
    edge_files = config.edge_files

    for ef in edge_files.split(','):
        edge_type, edge_file_or_dir = ef.split(":")
        graph.load_graph_from_files(os.path.join(edge_path, edge_file_or_dir), edge_type, config.symmetry)

    graph.load_node_feat(os.path.join(edge_path, config.node_types_file))
    graph.start_service(max_workers=12)

