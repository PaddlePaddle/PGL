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

from testsuite import create_random_graph


class GraphTest(unittest.TestCase):
    def test_check_to_tensor_to_numpy(self):
        num_nodes = 5
        dim = 4
        edges = [(0, 1), (1, 2), (3, 4)]
        nfeat = np.random.randn(num_nodes, dim)
        efeat = np.random.randn(len(edges), dim)

        g1 = pgl.Graph(
            edges=edges,
            num_nodes=num_nodes,
            node_feat={'nfeat': nfeat},
            edge_feat={'efeat': efeat})
        # check inplace to tensor
        self.assertFalse(g1.is_tensor())

        g2 = g1.tensor(inplace=False)
        g3 = g2.numpy(inplace=False)

        self.assertFalse(g1.is_tensor())
        self.assertTrue(g2.is_tensor())
        self.assertFalse(g3.is_tensor())

    def test_build_graph(self):

        num_nodes = 5
        dim = 4
        edges = [(0, 1), (1, 2), (3, 4)]
        nfeat = np.random.randn(num_nodes, dim)
        efeat = np.random.randn(len(edges), dim)

        g1 = pgl.Graph(
            edges=edges,
            num_nodes=num_nodes,
            node_feat={'nfeat': nfeat},
            edge_feat={'efeat': efeat})

    def test_build_tensor_graph(self):
        num_nodes = paddle.to_tensor(5)
        e = np.array([(0, 1), (1, 2), (3, 4)])
        edges = paddle.to_tensor(e)

        g2 = pgl.Graph(edges=edges, num_nodes=num_nodes)

    def test_build_graph_without_num_nodes(self):
        e = np.array([(0, 1), (1, 2), (3, 4)])
        edges = paddle.to_tensor(e)

        g2 = pgl.Graph(edges=edges)

    def test_neighbors(self):

        num_nodes = 5
        edges = [(0, 1), (0, 2), (1, 2), (3, 4)]
        g1 = pgl.Graph(edges=edges, num_nodes=num_nodes)

        pred, pred_eid = g1.predecessor(return_eids=True)
        self.assertEqual(len(pred), num_nodes)
        self.assertEqual(len(pred_eid), num_nodes)

        self.assertEqual(set(pred[0]), set([]))
        self.assertEqual(set(pred[1]), set([0]))
        self.assertEqual(set(pred[2]), set([0, 1]))
        self.assertEqual(set(pred[3]), set([]))
        self.assertEqual(set(pred[4]), set([3]))

        succ, succ_eid = g1.successor(return_eids=True)
        self.assertEqual(len(succ), num_nodes)
        self.assertEqual(len(succ_eid), num_nodes)

        self.assertEqual(set(succ[0]), set([1, 2]))
        self.assertEqual(set(succ[1]), set([2]))
        self.assertEqual(set(succ[2]), set([]))
        self.assertEqual(set(succ[3]), set([4]))
        self.assertEqual(set(succ[4]), set([]))

    def test_check_degree(self):
        """Check the degree
        """
        num_nodes = 5
        edges = [(0, 1), (1, 2), (3, 4)]
        g1 = pgl.Graph(edges=edges, num_nodes=num_nodes)
        indegree = np.array([0, 1, 1, 0, 1], dtype="int32")
        outdegree = np.array([1, 1, 0, 1, 0], dtype="int32")
        # check degree in numpy

        res = g1.indegree()
        self.assertTrue(np.all(res == indegree))

        # get degree from specific nodes
        res = g1.indegree(nodes=[1, 2, 3])
        self.assertTrue(np.all(res == indegree[[1, 2, 3]]))

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

            num_nodes = np.random.randint(low=2, high=10)
            edges = np.random.randint(
                low=1,
                high=num_nodes,
                size=[np.random.randint(
                    low=1, high=10), 2])
            nfeat = np.random.randn(num_nodes, dim)
            efeat = np.random.randn(len(edges), dim)

            g = pgl.Graph(
                edges=edges,
                num_nodes=num_nodes,
                node_feat={'nfeat': nfeat},
                edge_feat={'efeat': efeat})
            glist.append(g)
        # Merge Graph
        multi_graph = pgl.Graph.disjoint(glist)
        # Check Graph Index
        node_index = [np.ones(g.num_nodes) * n for n, g in enumerate(glist)]
        edge_index = [np.ones(g.num_edges) * n for n, g in enumerate(glist)]
        node_index = np.concatenate(node_index)
        edge_index = np.concatenate(edge_index)
        self.assertTrue(np.all(node_index == multi_graph.graph_node_id))
        self.assertTrue(np.all(edge_index == multi_graph.graph_edge_id))

        multi_graph.tensor()
        self.assertTrue(
            np.all(node_index == multi_graph.graph_node_id.numpy()))
        self.assertTrue(
            np.all(edge_index == multi_graph.graph_edge_id.numpy()))

        # testing for jointing One Graph
        multi_graph = pgl.Graph.disjoint([glist[0]])
        self.assertEqual(multi_graph.node_feat['nfeat'].shape,
                         (glist[0].num_nodes, dim))

    def test_disjoint_tensor_graph(self):
        glist = []
        dim = 4
        for i in range(5):
            num_nodes = np.random.randint(low=2, high=10)
            edges = np.random.randint(
                low=1,
                high=num_nodes,
                size=[np.random.randint(
                    low=1, high=10), 2])
            nfeat = np.random.randn(num_nodes, dim)
            efeat = np.random.randn(len(edges), dim)

            g = pgl.Graph(
                edges=paddle.to_tensor(edges),
                num_nodes=paddle.to_tensor(num_nodes),
                node_feat={"nfeat": paddle.to_tensor(nfeat)},
                edge_feat={"efeat": paddle.to_tensor(efeat)})
            glist.append(g)
        # Merge Graph
        multi_graph = pgl.Graph.disjoint(glist)

        # Check Graph Index
        node_index = [np.ones(g.num_nodes) * n for n, g in enumerate(glist)]
        edge_index = [np.ones(g.num_edges) * n for n, g in enumerate(glist)]
        node_index = np.concatenate(node_index)
        edge_index = np.concatenate(edge_index)

        self.assertTrue(
            np.all(node_index == multi_graph.graph_node_id.numpy()))
        self.assertTrue(
            np.all(edge_index == multi_graph.graph_edge_id.numpy()))

        multi_graph.numpy()
        self.assertTrue(np.all(node_index == multi_graph.graph_node_id))
        self.assertTrue(np.all(edge_index == multi_graph.graph_edge_id))

        # testing for jointing One Graph
        multi_graph = pgl.Graph.disjoint([glist[0]])
        self.assertEqual(multi_graph.node_feat['nfeat'].shape,
                         [int(glist[0].num_nodes), dim])

    def test_dump_numpy_load_tensor(self):

        path = './tmp'
        glist = []
        dim = 4
        num_nodes = 10
        edges = np.random.randint(
            low=1, high=num_nodes,
            size=[np.random.randint(
                low=2, high=10), 2])
        nfeat = np.random.randn(num_nodes, dim)
        efeat = np.random.randn(len(edges), dim)

        g = pgl.Graph(
            edges=edges,
            num_nodes=num_nodes,
            node_feat={'nfeat': nfeat},
            edge_feat={'efeat': efeat})

        in_before = g.indegree()
        g.outdegree()
        # Merge Graph
        g.dump(path)
        g2 = pgl.Graph.load(path)
        in_after = g2.indegree()
        for a, b in zip(in_before, in_after):
            self.assertEqual(a, b)

        del g2
        del in_after
        import shutil
        shutil.rmtree(path)

    def test_dump_tensor_load_numpy(self):
        path = './tmp'
        glist = []
        dim = 4
        num_nodes = 10
        edges = np.random.randint(
            low=1, high=num_nodes,
            size=[np.random.randint(
                low=2, high=10), 2])
        nfeat = np.random.randn(num_nodes, dim)
        efeat = np.random.randn(len(edges), dim)

        g = pgl.Graph(
            edges=edges,
            num_nodes=num_nodes,
            node_feat={'nfeat': nfeat},
            edge_feat={'efeat': efeat})

        in_before = g.indegree()
        g.outdegree()
        g.tensor()

        # Merge Graph
        g.dump(path)
        g2 = pgl.Graph.load(path)
        in_after = g2.indegree()
        for a, b in zip(in_before, in_after):
            self.assertEqual(a, b)

        del g2
        del in_after
        import shutil
        shutil.rmtree(path)

    def test_send_func(self):

        num_nodes = 4
        dim = 4
        edges = [(0, 1), (1, 2), (2, 3)]
        nfeat = np.arange(num_nodes).reshape(-1, 1)
        efeat = np.arange(len(edges)).reshape(-1, 1)

        target_src = np.array([0, 1, 2]).reshape(-1, 1)
        target_dst = np.array([1, 2, 3]).reshape(-1, 1)
        target_edge = np.array([0, 1, 2]).reshape(-1, 1)

        g1 = pgl.Graph(
            edges=edges,
            num_nodes=num_nodes,
            node_feat={'nfeat': nfeat},
            edge_feat={'efeat': efeat})

        g1.tensor()
        src_copy = lambda sf, df, ef: {"h": sf["h"], "e": ef["e"]}
        dst_copy = lambda sf, df, ef: {"h": df["h"], "e": ef["e"]}
        both_copy = lambda sf, df, ef: {"sh": sf["h"], "dh": df["h"], "e": ef["e"]}

        msg = g1.send(
            src_copy,
            src_feat={"h": g1.node_feat["nfeat"]},
            edge_feat={'e': g1.edge_feat['efeat']})
        self.assertTrue((msg['h'].numpy() == target_src).all())
        self.assertTrue((msg['e'].numpy() == target_edge).all())

        msg = g1.send(
            dst_copy,
            dst_feat={"h": g1.node_feat["nfeat"]},
            edge_feat={'e': g1.edge_feat['efeat']})
        self.assertTrue((msg['h'].numpy() == target_dst).all())
        self.assertTrue((msg['e'].numpy() == target_edge).all())

        msg = g1.send(
            both_copy,
            node_feat={"h": g1.node_feat["nfeat"]},
            edge_feat={'e': g1.edge_feat['efeat']})
        self.assertTrue((msg['sh'].numpy() == target_src).all())
        self.assertTrue((msg['dh'].numpy() == target_dst).all())
        self.assertTrue((msg['e'].numpy() == target_edge).all())

    def test_send_recv_func(self):
        np.random.seed(0)
        num_nodes = 5
        dim = 4
        edges = [(0, 1), (1, 2), (3, 4), (4, 1), (1, 0)]

        nfeat = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6],
                          [4, 5, 6, 7], [5, 6, 7, 8]])

        ground = np.array([[2, 3, 4, 5], [6, 8, 10, 12], [2, 3, 4, 5],
                           [0, 0, 0, 0], [4, 5, 6, 7]])

        g = pgl.Graph(
            edges=edges, num_nodes=num_nodes, node_feat={'nfeat': nfeat})

        g.tensor()

        output = g.send_recv(g.node_feat['nfeat'], reduce_func="sum")
        output = output.numpy()

        self.assertTrue((ground == output).all())

    def test_send_and_recv(self):
        np.random.seed(0)
        num_nodes = 5
        dim = 4
        edges = [(0, 1), (1, 2), (3, 4), (4, 1), (1, 0)]

        nfeat = np.array(
            [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7],
             [5, 6, 7, 8]],
            dtype="float32")

        ground = np.array(
            [[1, 2, 3, 4], [2, 3, 4, 5], [4, 5, 6, 7], [5, 6, 7, 8],
             [2, 3, 4, 5]],
            dtype="float32")

        recv_ground = np.array(
            [[2., 3., 4., 5.], [6., 8., 10., 12.], [2., 3., 4., 5.],
             [0., 0., 0., 0.], [4., 5., 6., 7.]],
            dtype="float32")

        g = pgl.Graph(
            edges=edges, num_nodes=num_nodes, node_feat={'nfeat': nfeat})

        g.tensor()

        def send_func1(src_feat, dst_feat, edge_feat):
            return src_feat

        def send_func2(src_feat, dst_feat, edge_feat):
            return {'h': src_feat['h']}

        def reduce_func(msg):
            return msg.reduce_sum(msg['h'])

        # test send_func1
        msg1 = g.send(send_func1, src_feat={'h': g.node_feat['nfeat']})
        _msg = msg1['h'].numpy()
        self.assertTrue((ground == _msg).all())

        output = g.recv(reduce_func, msg1)
        output = output.numpy()
        self.assertTrue((recv_ground == output).all())

        # test send_func2
        msg2 = g.send(send_func1, src_feat={'h': g.node_feat['nfeat']})
        _msg = msg2['h'].numpy()
        self.assertTrue((ground == _msg).all())

        output = g.recv(reduce_func, msg2)
        output = output.numpy()
        self.assertTrue((recv_ground == output).all())

    def test_node_iter(self):
        np_graph = create_random_graph().numpy()
        pd_graph = create_random_graph().tensor()

        for graph in [np_graph, pd_graph]:
            assert len(graph.nodes) == graph.num_nodes
            for shuffle in [True, False]:
                for batch_data in graph.node_batch_iter(
                        batch_size=3, shuffle=shuffle):
                    break


if __name__ == "__main__":
    unittest.main()
