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
"""PGL"""
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor

import grpc

import numpy as np
from pgl.contrib.distributed import graph_service_pb2
from pgl.contrib.distributed import graph_service_pb2_grpc


from pgl.utils.logger import log
from collections import defaultdict
import time
import sys
import os
SampleNeighborsReply = graph_service_pb2.SampleNeighborsReply
SampleNeighborsRequest = graph_service_pb2.SampleNeighborsRequest
NodeList = graph_service_pb2.NodeList

def grpc_server_on(channel):
    TIMEOUT_SEC=15
    try:
        grpc.channel_ready_future(channel).result(timeout=TIMEOUT_SEC)
        return True
    except grpc.FutureTimeoutError:
        return False

class ServerEndpoints(object):
    def __init__(self, server_endpoint):
        cc = 0
        while not os.path.exists(server_endpoint):
            if cc % 50 == 0:
                log.info("Server haven't started yet")
            cc += 1
            time.sleep(5)

        time.sleep(10)
        self.endpoint_list = []
        with open(server_endpoint) as f:
            for line in f:
                line = line.strip().split(',')
                self.endpoint_list.append(line)

    def __len__(self):
        return len(self.endpoint_list)

    def __getitem__(self, key):
        if len(self.endpoint_list[key]) == 1:
            return self.endpoint_list[key][0]
        else:
            return np.random.choice(self.endpoint_list[key])


class ServerChannel(object):
    def __init__(self, server_endpoints):
        self.server_channel = [] 
        for server_index in range(len(server_endpoints)):
            endpoints = server_endpoints.endpoint_list[server_index]
            channels = []
            for endpoint in endpoints:
                channel = grpc.insecure_channel(endpoint)
                while True:
                    flag = grpc_server_on(channel)
                    if not flag:
                        log.info(endpoint, " is not ready. sleep 3s")
                        time.sleep(3)
                    else:
                        break
                channels.append(channel)
            self.server_channel.append(channels)

    def __len__(self):
        return len(self.server_channel)

    def __getitem__(self, key):
        if len(self.server_channel[key]) == 1:
            return self.server_channel[key][0]
        else:
            return np.random.choice(self.server_channel[key])



class DistCPUGraphClient(object):
    def __init__(self, server_endpoint, shard_num=1000):
        self.shard_num = 1000
        self.server_endpoint = ServerEndpoints(server_endpoint)
        self.server_channel = ServerChannel(self.server_endpoint)
        log.info("Server ips %s" % (self.server_endpoint.endpoint_list, ))
        self.executor = ThreadPoolExecutor()

    def select_server(self, nid):
        return nid % self.shard_num % len(self.server_endpoint)

    def sample_predecessor(self, nodes, max_degree, edge_type):

        start = time.time()
        node_dict = defaultdict(lambda : []) 
        for n, node in enumerate(nodes): 
            node_dict[self.select_server(node)].append((n, node))
        
        output = [[] for _ in nodes]

        def processor(server_index, batch_nodes):
            nid, batch_nodes = zip(*batch_nodes)
            stub = graph_service_pb2_grpc.GraphServiceStub(self.server_channel[server_index])
            response = stub.SampleNeighbors(SampleNeighborsRequest(max_size=max_degree,
                                                sample_nodes=NodeList(nodeid=batch_nodes), edge_type=edge_type))
            for n, neighbor in zip(nid, response.neighbors):
                output[n].extend(neighbor.nodeid)


        result = []
        for server_index, batch_nodes in node_dict.items():
            res = self.executor.submit(processor, server_index, batch_nodes)
            result.append(res)

        futures.wait(result)

        return output

    def sample_successor(self, nodes, max_degree, edge_type):
        return self.sample_predecessor(nodes, max_degree, edge_type)


    def get_node_feat(self, nodes):
        start = time.time()
 
        node_dict = defaultdict(lambda : []) 
        for node in nodes: 
            node_dict[self.select_server(node)].append(node)
        
        output = defaultdict(lambda : "")

        def processor(server_index, batch_nodes):
            stub = graph_service_pb2_grpc.GraphServiceStub(self.server_channel[server_index])
            response = stub.GetNodeFeat(graph_service_pb2.GetNodeFeatRequest(nodes=NodeList(nodeid=batch_nodes)))
            for node, value in zip(response.nodes.nodeid, response.value.split("\n")):
                output[node] = value

        result = []
        for server_index, batch_nodes in node_dict.items():
            res = self.executor.submit(processor, server_index, batch_nodes)
            result.append(res)
        futures.wait(result)
        return [output[n] for n in nodes]
                
    def node_batch_iter(self, batch_size, node_type, shuffle=True, rank=0, nrank=1): 

        number_of_server = len(self.server_channel)
        servers = np.arange(number_of_server)
        if shuffle:
            np.random.shuffle(servers)

        for server_index in servers:
            stub = graph_service_pb2_grpc.GraphServiceStub(self.server_channel[server_index])
            for res in stub.NodeBatchIter(graph_service_pb2.SampleNodesRequest(batch_size=batch_size, shuffle=shuffle, rank=rank, nrank=nrank, node_type=node_type)):
                yield res.sample_nodes.nodeid

    def sample_nodes(self, nodes, neg_num, neg_sample_type):
        start = time.time()
        node_dict = defaultdict(lambda : []) 
        for n, node in enumerate(nodes): 
            node_dict[self.select_server(node)].append((n, node))
        
        output = np.zeros(shape=[len(nodes), neg_num], dtype="int64") 

        def processor(server_index, batch_nodes):
            nid, batch_nodes = zip(*batch_nodes)
            stub = graph_service_pb2_grpc.GraphServiceStub(self.server_channel[server_index])
            response = stub.NegSampleNodes(graph_service_pb2.NegSampleNodesRequest(
                                                nodeid=NodeList(nodeid=batch_nodes),
                                                neg_num=neg_num,
                                                neg_sample_type=neg_sample_type))

            negs = np.array(response.sample_nodes.nodeid).reshape([-1, neg_num])
            output[nid, :] = negs

        result = []
        for server_index, batch_nodes in node_dict.items():
            res = self.executor.submit(processor, server_index, batch_nodes)
            result.append(res)

        futures.wait(result)

        end = time.time()
        #self.profile("sample_nodes", end - start)
        return output

    def get_all_types(self):
        number_of_server = len(self.server_channel)
        server_index  = np.random.randint(0, number_of_server)
        stub = graph_service_pb2_grpc.GraphServiceStub(self.server_channel[server_index])
        response = stub.NodeTypes(graph_service_pb2.NodeTypesRequest())
        return response.node_types

if __name__ == "__main__":
    graph = DistCPUGraphClient("server_endpoints", shard_num=1000)
    

