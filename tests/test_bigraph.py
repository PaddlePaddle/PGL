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

from testsuite import create_random_bigraph


class BiGraphTest(unittest.TestCase):
    def test_check_to_tensor_to_numpy(self):

        src_num_nodes = 4
        dst_num_nodes = 5
        dim = 4
        edges = [(0, 1), (1, 2), (3, 4)]
        src_nfeat = np.random.randn(src_num_nodes, dim)
        dst_nfeat = np.random.randn(dst_num_nodes, dim)
        efeat = np.random.randn(len(edges), dim)

        g1 = pgl.BiGraph(
            edges=edges,
            src_num_nodes=src_num_nodes,
            dst_num_nodes=dst_num_nodes,
            src_node_feat={'src_nfeat': src_nfeat},
            dst_node_feat={'dst_nfeat': dst_nfeat},
            edge_feat={'efeat': efeat})
        # check inplace to tensor
        self.assertFalse(g1.is_tensor())

        g2 = g1.tensor(inplace=False)
        g3 = g2.numpy(inplace=False)

        self.assertFalse(g1.is_tensor())
        self.assertTrue(g2.is_tensor())
        self.assertFalse(g3.is_tensor())

    def test_build_graph(self):

        src_num_nodes = 4
        dst_num_nodes = 5
        dim = 4
        edges = [(0, 1), (1, 2), (3, 4)]
        src_nfeat = np.random.randn(src_num_nodes, dim)
        dst_nfeat = np.random.randn(dst_num_nodes, dim)
        efeat = np.random.randn(len(edges), dim)

        g1 = pgl.BiGraph(
            edges=edges,
            src_num_nodes=src_num_nodes,
            dst_num_nodes=dst_num_nodes,
            src_node_feat={'src_nfeat': src_nfeat},
            dst_node_feat={'dst_nfeat': dst_nfeat},
            edge_feat={'efeat': efeat})

    def test_build_tensor_graph(self):
        src_num_nodes = paddle.to_tensor(4)
        dst_num_nodes = paddle.to_tensor(5)
        e = np.array([(0, 1), (1, 2), (3, 4)])
        edges = paddle.to_tensor(e)

        g2 = pgl.BiGraph(
            edges=edges,
            src_num_nodes=src_num_nodes,
            dst_num_nodes=dst_num_nodes)

    def test_build_graph_without_num_nodes(self):
        e = np.array([(0, 1), (1, 2), (3, 4)])
        edges = paddle.to_tensor(e)

        g2 = pgl.BiGraph(edges=edges)

    def test_check_degree(self):
        """Check the de
        """
        src_num_nodes = 4
        dst_num_nodes = 5
        edges = [(0, 1), (1, 2), (3, 4)]
        g1 = pgl.BiGraph(
            edges=edges,
            src_num_nodes=src_num_nodes,
            dst_num_nodes=dst_num_nodes)
        indegree = np.array([0, 1, 1, 0, 1], dtype="int32")
        outdegree = np.array([1, 1, 0, 1], dtype="int32")
        # check degree in numpy

        res = g1.indegree()
        self.assertTrue(np.all(res == indegree))

        # get degree from specific nodes
        res = g1.indegree(nodes=[1, 2, 3, 4])
        self.assertTrue(np.all(res == indegree[[1, 2, 3, 4]]))

        res = g1.outdegree()
        self.assertTrue(np.all(res == outdegree))

        # get degree from specific nodes
        res = g1.outdegree(nodes=[1, 2, 3])
        self.assertTrue(np.all(res == outdegree[[1, 2, 3]]))

        # check degree in Tensor 
        g1.tensor()

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

    def test_disjoint_graph(self):
        glist = []
        dim = 4
        for i in range(5):

            src_num_nodes = np.random.randint(low=2, high=10)
            dst_num_nodes = np.random.randint(low=2, high=10)
            edges_size = np.random.randint(low=1, high=10)
            edges_src = np.random.randint(
                low=1, high=src_num_nodes, size=[edges_size, 1])

            edges_dst = np.random.randint(
                low=1, high=dst_num_nodes, size=[edges_size, 1])

            edges = np.hstack([edges_src, edges_dst])

            src_nfeat = np.random.randn(src_num_nodes, dim)
            dst_nfeat = np.random.randn(dst_num_nodes, dim)
            efeat = np.random.randn(len(edges), dim)

            g = pgl.BiGraph(
                edges=edges,
                src_num_nodes=src_num_nodes,
                dst_num_nodes=dst_num_nodes,
                src_node_feat={'nfeat': src_nfeat},
                dst_node_feat={'nfeat': dst_nfeat},
                edge_feat={'efeat': efeat})
            glist.append(g)
        # Merge Graph
        b_graph = pgl.BiGraph.batch(glist)
        multi_graph = pgl.BiGraph.disjoint(glist)
        # Check Graph Index
        src_node_index = [
            np.ones(g.src_num_nodes) * n for n, g in enumerate(glist)
        ]
        dst_node_index = [
            np.ones(g.dst_num_nodes) * n for n, g in enumerate(glist)
        ]
        edge_index = [np.ones(g.num_edges) * n for n, g in enumerate(glist)]
        src_node_index = np.concatenate(src_node_index)
        dst_node_index = np.concatenate(dst_node_index)
        edge_index = np.concatenate(edge_index)
        self.assertTrue(
            np.all(src_node_index == multi_graph.graph_src_node_id))
        self.assertTrue(
            np.all(dst_node_index == multi_graph.graph_dst_node_id))
        self.assertTrue(np.all(edge_index == multi_graph.graph_edge_id))

        multi_graph.tensor()
        self.assertTrue(
            np.all(src_node_index == multi_graph.graph_src_node_id.numpy()))
        self.assertTrue(
            np.all(dst_node_index == multi_graph.graph_dst_node_id.numpy()))
        self.assertTrue(
            np.all(edge_index == multi_graph.graph_edge_id.numpy()))

        # testing for jointing One Graph
        multi_graph = pgl.BiGraph.disjoint([glist[0]])
        self.assertEqual(multi_graph.src_node_feat['nfeat'].shape,
                         (glist[0].src_num_nodes, dim))
        self.assertEqual(multi_graph.dst_node_feat['nfeat'].shape,
                         (glist[0].dst_num_nodes, dim))

    def test_disjoint_tensor_graph(self):
        glist = []
        dim = 4
        for i in range(5):

            src_num_nodes = np.random.randint(low=2, high=10)
            dst_num_nodes = np.random.randint(low=2, high=10)
            edges_size = np.random.randint(low=1, high=10)
            edges_src = np.random.randint(
                low=1, high=src_num_nodes, size=[edges_size, 1])

            edges_dst = np.random.randint(
                low=1, high=dst_num_nodes, size=[edges_size, 1])

            edges = np.hstack([edges_src, edges_dst])

            src_nfeat = np.random.randn(src_num_nodes, dim)
            dst_nfeat = np.random.randn(dst_num_nodes, dim)
            efeat = np.random.randn(len(edges), dim)

            g = pgl.BiGraph(
                edges=paddle.to_tensor(edges),
                src_num_nodes=paddle.to_tensor(src_num_nodes),
                dst_num_nodes=paddle.to_tensor(dst_num_nodes),
                src_node_feat={'nfeat': paddle.to_tensor(src_nfeat)},
                dst_node_feat={'nfeat': paddle.to_tensor(dst_nfeat)},
                edge_feat={'efeat': paddle.to_tensor(efeat)})
            glist.append(g)
        # Merge Graph
        multi_graph = pgl.BiGraph.disjoint(glist)
        # Check Graph Index
        src_node_index = [
            np.ones(g.src_num_nodes) * n for n, g in enumerate(glist)
        ]
        dst_node_index = [
            np.ones(g.dst_num_nodes) * n for n, g in enumerate(glist)
        ]
        edge_index = [np.ones(g.num_edges) * n for n, g in enumerate(glist)]
        src_node_index = np.concatenate(src_node_index)
        dst_node_index = np.concatenate(dst_node_index)
        edge_index = np.concatenate(edge_index)
        self.assertTrue(
            np.all(src_node_index == multi_graph.graph_src_node_id.numpy()))
        self.assertTrue(
            np.all(dst_node_index == multi_graph.graph_dst_node_id.numpy()))
        self.assertTrue(
            np.all(edge_index == multi_graph.graph_edge_id.numpy()))

        multi_graph.tensor()
        self.assertTrue(
            np.all(src_node_index == multi_graph.graph_src_node_id.numpy()))
        self.assertTrue(
            np.all(dst_node_index == multi_graph.graph_dst_node_id.numpy()))
        self.assertTrue(
            np.all(edge_index == multi_graph.graph_edge_id.numpy()))

        multi_graph = pgl.BiGraph.disjoint([glist[0]])
        self.assertEqual(multi_graph.src_node_feat['nfeat'].shape,
                         [int(glist[0].src_num_nodes), dim])
        self.assertEqual(multi_graph.dst_node_feat['nfeat'].shape,
                         [int(glist[0].dst_num_nodes), dim])

    def test_dump_numpy_load_numpy(self):

        path = './tmp'
        dim = 4
        src_num_nodes = np.random.randint(low=2, high=10)
        dst_num_nodes = np.random.randint(low=2, high=10)
        edges_size = np.random.randint(low=1, high=10)
        edges_src = np.random.randint(
            low=1, high=src_num_nodes, size=[edges_size, 1])

        edges_dst = np.random.randint(
            low=1, high=dst_num_nodes, size=[edges_size, 1])

        edges = np.hstack([edges_src, edges_dst])

        src_nfeat = np.random.randn(src_num_nodes, dim)
        dst_nfeat = np.random.randn(dst_num_nodes, dim)
        efeat = np.random.randn(len(edges), dim)

        g = pgl.BiGraph(
            edges=paddle.to_tensor(edges),
            src_num_nodes=paddle.to_tensor(src_num_nodes),
            dst_num_nodes=paddle.to_tensor(dst_num_nodes),
            src_node_feat={'nfeat': paddle.to_tensor(src_nfeat)},
            dst_node_feat={'nfeat': paddle.to_tensor(dst_nfeat)},
            edge_feat={'efeat': paddle.to_tensor(efeat)})

        in_before = g.indegree()
        g.outdegree()
        # Merge Graph
        g.dump(path)
        g2 = pgl.BiGraph.load(path)
        in_after = g2.indegree()
        for a, b in zip(in_before, in_after):
            self.assertEqual(a, b)

        del g2
        del in_after
        import shutil
        shutil.rmtree(path)

    def test_dump_tensor_load_tensor(self):
        path = './tmp'
        dim = 4
        src_num_nodes = np.random.randint(low=2, high=10)
        dst_num_nodes = np.random.randint(low=2, high=10)
        edges_size = np.random.randint(low=1, high=10)
        edges_src = np.random.randint(
            low=1, high=src_num_nodes, size=[edges_size, 1])

        edges_dst = np.random.randint(
            low=1, high=dst_num_nodes, size=[edges_size, 1])

        edges = np.hstack([edges_src, edges_dst])

        src_nfeat = np.random.randn(src_num_nodes, dim)
        dst_nfeat = np.random.randn(dst_num_nodes, dim)
        efeat = np.random.randn(len(edges), dim)

        g = pgl.BiGraph(
            edges=paddle.to_tensor(edges),
            src_num_nodes=paddle.to_tensor(src_num_nodes),
            dst_num_nodes=paddle.to_tensor(dst_num_nodes),
            src_node_feat={'nfeat': paddle.to_tensor(src_nfeat)},
            dst_node_feat={'nfeat': paddle.to_tensor(dst_nfeat)},
            edge_feat={'efeat': paddle.to_tensor(efeat)})

        in_before = g.indegree()
        g.outdegree()
        g.tensor()

        # Merge Graph
        g.dump(path)
        g2 = pgl.BiGraph.load(path)
        in_after = g2.indegree()
        for a, b in zip(in_before, in_after):
            self.assertEqual(a, b)

        del g2
        del in_after
        import shutil
        shutil.rmtree(path)

    def test_send_func(self):

        src_num_nodes = 4
        dst_num_nodes = 5
        edges = [(0, 1), (1, 2), (3, 4)]
        src_nfeat = np.arange(src_num_nodes).reshape(-1, 1)
        dst_nfeat = np.arange(dst_num_nodes).reshape(-1, 1)
        efeat = np.arange(len(edges)).reshape(-1, 1)

        g1 = pgl.BiGraph(
            edges=edges,
            src_num_nodes=src_num_nodes,
            dst_num_nodes=dst_num_nodes,
            src_node_feat={'src_nfeat': src_nfeat},
            dst_node_feat={'dst_nfeat': dst_nfeat},
            edge_feat={'efeat': efeat})

        target_src = np.array([0, 1, 3]).reshape(-1, 1)
        target_dst = np.array([1, 2, 4]).reshape(-1, 1)
        target_edge = np.array([0, 1, 2]).reshape(-1, 1)

        g1.tensor()
        src_copy = lambda sf, df, ef: {"h": sf["h"], "e": ef["e"]}
        dst_copy = lambda sf, df, ef: {"h": df["h"], "e": ef["e"]}
        both_copy = lambda sf, df, ef: {"sh": sf["h"], "dh": df["h"], "e": ef["e"]}

        msg = g1.send(
            src_copy,
            src_feat={"h": g1.src_node_feat["src_nfeat"]},
            edge_feat={'e': g1.edge_feat['efeat']})
        self.assertTrue((msg['h'].numpy() == target_src).all())
        self.assertTrue((msg['e'].numpy() == target_edge).all())

        msg = g1.send(
            dst_copy,
            dst_feat={"h": g1.dst_node_feat["dst_nfeat"]},
            edge_feat={'e': g1.edge_feat['efeat']})
        self.assertTrue((msg['h'].numpy() == target_dst).all())
        self.assertTrue((msg['e'].numpy() == target_edge).all())

        msg = g1.send(
            both_copy,
            src_feat={"h": g1.src_node_feat["src_nfeat"]},
            dst_feat={"h": g1.dst_node_feat["dst_nfeat"]},
            edge_feat={'e': g1.edge_feat['efeat']})
        self.assertTrue((msg['sh'].numpy() == target_src).all())
        self.assertTrue((msg['dh'].numpy() == target_dst).all())
        self.assertTrue((msg['e'].numpy() == target_edge).all())

    def test_send_recv_func(self):

        np.random.seed(0)
        src_num_nodes = 5
        dst_num_nodes = 4
        edges = [(0, 1), (1, 2), (3, 3), (4, 1), (1, 0)]
        src_nfeat = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6],
                              [4, 5, 6, 7], [5, 6, 7, 8]])

        ground = np.array([[2, 3, 4, 5], [6, 8, 10, 12], [2, 3, 4, 5],
                           [4, 5, 6, 7]])

        g = pgl.BiGraph(
            edges=edges,
            src_num_nodes=src_num_nodes,
            dst_num_nodes=dst_num_nodes,
            src_node_feat={'src_nfeat': src_nfeat}, )
        g.tensor()

        output = g.send_recv(g.src_node_feat['src_nfeat'], reduce_func="sum")
        output = output.numpy()

        self.assertTrue((ground == output).all())

    def test_send_and_recv(self):
        np.random.seed(0)
        src_num_nodes = 5
        dst_num_nodes = 4
        edges = [(0, 1), (1, 2), (3, 3), (4, 1), (1, 0)]
        src_nfeat = np.array(
            [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7],
             [5, 6, 7, 8]],
            dtype="float32")

        dst_nfeat = np.array(
            [[2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]],
            dtype="float32")

        src_ground = np.array(
            [[1, 2, 3, 4], [2, 3, 4, 5], [4, 5, 6, 7], [5, 6, 7, 8],
             [2, 3, 4, 5]],
            dtype="float32")

        src_recv = np.array([[2, 3, 4, 5], [6, 8, 10, 12], [2, 3, 4, 5],
                             [4, 5, 6, 7]])

        dst_ground = np.array(
            [[3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [3, 4, 5, 6],
             [2, 3, 4, 5]],
            dtype="float32")

        dst_recv = np.array([[3, 4, 5, 6], [6, 8, 10, 12], [0, 0, 0, 0],
                             [5, 6, 7, 8], [3, 4, 5, 6]])

        g = pgl.BiGraph(
            edges=edges,
            src_num_nodes=src_num_nodes,
            dst_num_nodes=dst_num_nodes,
            src_node_feat={'src_nfeat': src_nfeat},
            dst_node_feat={'dst_nfeat': dst_nfeat})

        g.tensor()

        def send_func1(src_feat, dst_feat, edge_feat):
            return src_feat

        def send_func2(src_feat, dst_feat, edge_feat):
            return {'h': src_feat['h']}

        def reduce_func(msg):
            return msg.reduce_sum(msg['h'])

        # test send_func1
        msg1 = g.send(send_func1, src_feat={'h': g.src_node_feat['src_nfeat']})
        _msg = msg1['h'].numpy()
        self.assertTrue((src_ground == _msg).all())

        output = g.recv(reduce_func, msg1)
        output = output.numpy()
        self.assertTrue((src_recv == output).all())

        # test send_func2
        msg2 = g.send(send_func2, src_feat={'h': g.src_node_feat['src_nfeat']})
        _msg = msg2['h'].numpy()
        self.assertTrue((src_ground == _msg).all())

        output = g.recv(reduce_func, msg2)
        output = output.numpy()
        self.assertTrue((src_recv == output).all())

        def send_func1_d(src_feat, dst_feat, edge_feat):
            return dst_feat

        def send_func2_d(src_feat, dst_feat, edge_feat):
            return {'h': dst_feat['h']}

        def reduce_func_d(msg):
            return msg.reduce_sum(msg['h'])

        # test send_func1
        msg1 = g.send(
            send_func1_d, dst_feat={'h': g.dst_node_feat['dst_nfeat']})
        _msg = msg1['h'].numpy()
        self.assertTrue((dst_ground == _msg).all())

        output = g.recv(reduce_func_d, msg1, recv_mode="src")
        output = output.numpy()
        self.assertTrue((dst_recv == output).all())

        # test send_func2
        msg2 = g.send(
            send_func2_d, dst_feat={'h': g.dst_node_feat['dst_nfeat']})
        _msg = msg2['h'].numpy()
        self.assertTrue((dst_ground == _msg).all())

        output = g.recv(reduce_func_d, msg2, recv_mode="src")
        output = output.numpy()
        self.assertTrue((dst_recv == output).all())

    def test_node_iter(self):
        np_graph = create_random_bigraph().numpy()
        pd_graph = create_random_bigraph().tensor()

        for graph in [np_graph, pd_graph]:
            assert len(graph.src_nodes) == graph.src_num_nodes
            assert len(graph.dst_nodes) == graph.dst_num_nodes
            for shuffle in [True, False]:
                for m in ["src_node", "dst_node"]:
                    for batch_data in graph.node_batch_iter(
                            batch_size=3, shuffle=shuffle, mode=m):
                        break


if __name__ == "__main__":
    unittest.main()
