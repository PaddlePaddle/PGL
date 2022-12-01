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

import os

import unittest
import numpy as np
import paddle
import pgl
import paddle.distributed as dist

from testsuite import create_random_graph


class TestDistGPUGraph(unittest.TestCase):
    def test_distributed_degree(self):
        paddle.distributed.init_parallel_env()
        num_nodes = 5
        edges = [(0, 1), (1, 2), (3, 4)]
        g1 = pgl.Graph(edges=edges, num_nodes=num_nodes)
        indegree = np.array([0, 1, 1, 0, 1], dtype="int32")
        outdegree = np.array([1, 1, 0, 1, 0], dtype="int32")
        g1 = pgl.DistGPUGraph(g1)

        res = g1.indegree().numpy()
        self.assertTrue(np.all(res == indegree))

        # get degree from specific nodes
        res = g1.indegree(nodes=paddle.to_tensor([1, 2, 3])).numpy()
        self.assertTrue(np.all(res == indegree[[1, 2, 3]]))

        res = g1.outdegree().numpy()
        self.assertTrue(np.all(res == outdegree))

        # get degree from specific nodes
        res = g1.outdegree(nodes=paddle.to_tensor([1, 2, 3])).numpy()
        self.assertTrue(np.all(res == outdegree[[1, 2, 3]]))

    def test_distributed_send_recv(self):
        paddle.distributed.init_parallel_env()
        # Test Send-Recv
        num_nodes = 5
        edges = [(0, 1), (1, 2), (3, 4), (4, 1), (1, 0)]

        nfeat = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6],
                          [4, 5, 6, 7], [5, 6, 7, 8]])

        ground = np.array([[2, 3, 4, 5], [6, 8, 10, 12], [2, 3, 4, 5],
                           [0, 0, 0, 0], [4, 5, 6, 7]])

        g = pgl.Graph(
            edges=edges, num_nodes=num_nodes, node_feat={'nfeat': nfeat})
        g = pgl.DistGPUGraph(g)

        output = g.send_recv(g.node_feat['nfeat'], reduce_func="sum")
        output = output.numpy()

        self.assertTrue((ground == output).all())

    def test_distributed_send_then_recv(self):
        paddle.distributed.init_parallel_env()
        # Test Send Then Recv
        num_nodes = 5
        edges = [(0, 1), (1, 2), (3, 4), (4, 1), (1, 0)]

        nfeat = np.array(
            [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7],
             [5, 6, 7, 8]],
            dtype="float32")

        recv_ground = np.array(
            [[2., 3., 4., 5.], [6., 8., 10., 12.], [2., 3., 4., 5.],
             [0., 0., 0., 0.], [4., 5., 6., 7.]],
            dtype="float32")

        g = pgl.Graph(
            edges=edges, num_nodes=num_nodes, node_feat={'nfeat': nfeat})

        g = pgl.DistGPUGraph(g)

        def send_func1(src_feat, dst_feat, edge_feat):
            return src_feat

        def send_func2(src_feat, dst_feat, edge_feat):
            return {'h': src_feat['h']}

        def reduce_func(msg):
            return msg.reduce_sum(msg['h'])

        # test send_func1
        msg1 = g.send(send_func1, src_feat={'h': g.node_feat['nfeat']})

        output = g.recv(reduce_func, msg1)
        output = output.numpy()
        self.assertTrue((recv_ground == output).all())

        # test send_func2
        msg2 = g.send(send_func1, src_feat={'h': g.node_feat['nfeat']})

        output = g.recv(reduce_func, msg2)
        output = output.numpy()
        self.assertTrue((recv_ground == output).all())

    def test_distributed_send_ue_recv(self):
        # test send_ue_recv
        paddle.distributed.init_parallel_env()
        num_nodes = 5
        edges = [(0, 1), (1, 2), (3, 4), (4, 1), (1, 0)]
        nfeat = np.array(
            [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7],
             [5, 6, 7, 8]],
            dtype="float32")
        efeat = np.array([1, 1, 1, 1, 1], dtype="float32")
        ue_recv_ground = np.array(
            [[3., 4., 5., 6.], [8., 10., 12., 14.], [3., 4., 5., 6.],
             [0., 0., 0., 0.], [5., 6., 7., 8.]],
            dtype="float32")
        g = pgl.Graph(
            edges=edges,
            num_nodes=num_nodes,
            node_feat={'nfeat': nfeat},
            edge_feat={'efeat': efeat})
        g = pgl.DistGPUGraph(g)
        output = g.send_ue_recv(g.node_feat['nfeat'], g.edge_feat['efeat'])
        output = output.numpy()
        self.assertTrue((ue_recv_ground == output).all())


if __name__ == "__main__":
    #  Test CUDA_VISIBLE_DEVICES=0,1 python -m paddle.distributed.launch test_dist_graph.py
    unittest.main()
