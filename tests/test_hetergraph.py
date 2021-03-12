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
import shutil


class HeterGraphTest(unittest.TestCase):

    def test_build_hetergraph(self):
        np.random.seed(1)
        dim = 4
        num_nodes = 15

        edges = {}
        # for test no successor
        edges['c2p'] = [(1, 4), (0, 5), (1, 9), (1, 8), (2, 8), (2, 5), (3, 6),
                        (3, 7), (3, 4), (3, 8)]
        edges['p2c'] = [(v, u) for u, v in edges['c2p']]
        edges['p2a'] = [(4, 10), (4, 11), (4, 12), (4, 14), (4, 13), (6, 12),
                        (6, 11), (6, 14), (7, 12), (7, 11), (8, 14), (9, 10)]
        edges['a2p'] = [(v, u) for u, v in edges['p2a']]

        node_types = ['c' for _ in range(4)] + \
                     ['p' for _ in range(6)] + \
                     ['a' for _ in range(5)]
        node_types = [(i, t) for i, t in enumerate(node_types)]

        nfeat = {'nfeat': np.random.randn(num_nodes, dim)}
        efeat = {}
        for etype, _edges in edges.items():
            efeat[etype] = {'efeat': np.random.randn(len(_edges), dim)}

        hg = pgl.HeterGraph(edges=edges,
                node_types=node_types,
                node_feat=nfeat,
                edge_feat=efeat)

        self.assertFalse(hg.is_tensor())
        self.assertEqual(hg.indegree(5), [2])
        self.assertEqual(hg.outdegree(4), [7])
        self.assertEqual(hg.outdegree(4, 'c2p'), [0])
        self.assertEqual(hg.successor('c2p', [4]).tolist(), [[]])
        self.assertEqual(hg.predecessor('a2p', [4]).tolist(), [[10,11,12,14,13]])
        print()
        #  print(hg.predecessor('a2p', [4]))
        for batch in hg.node_batch_iter(3):
            break

    def test_build_tensor_hetergraph(self):
        np.random.seed(1)
        dim = 4
        num_nodes = paddle.to_tensor(15)

        edges = {}
        # for test no successor
        c2p = [(1, 4), (0, 5), (1, 9), (1, 8), (2, 8), 
               (2, 5), (3, 6), (3, 7), (3, 4), (3, 8)]
        edges['c2p'] = paddle.to_tensor(np.array(c2p))
        p2c = [(v, u) for u, v in c2p]
        edges['p2c'] = paddle.to_tensor(np.array(p2c))

        p2a = [(4, 10), (4, 11), (4, 12), (4, 14), (4, 13), (6, 12),
                        (6, 11), (6, 14), (7, 12), (7, 11), (8, 14), (9, 10)]
        edges['p2a'] = paddle.to_tensor(np.array(p2a))
        a2p = [(v, u) for u, v in p2a]
        edges['a2p'] = paddle.to_tensor(np.array(a2p))

        node_types = ['c' for _ in range(4)] + \
                     ['p' for _ in range(6)] + \
                     ['a' for _ in range(5)]
        node_types = [(i, t) for i, t in enumerate(node_types)]

        hg = pgl.HeterGraph(edges=edges,
                node_types=node_types)

        self.assertTrue(hg.is_tensor())
        print()
        self.assertEqual(hg.indegree(paddle.to_tensor(5)).numpy(), np.array([2]))
        self.assertEqual(hg.indegree(paddle.to_tensor(5), 'c2p').numpy(), np.array([2]))
        self.assertEqual(hg.outdegree(paddle.to_tensor(4)).numpy(), np.array([7]))
        self.assertEqual(hg.outdegree(paddle.to_tensor(4), 'p2a').numpy(), np.array([5]))
        #  print(hg.outdegree(paddle.to_tensor(4)).numpy())
        for batch in hg.node_batch_iter(3, n_type='c'):
            break

    def test_tensor(self):
        np.random.seed(1)
        dim = 4
        num_nodes = 15

        edges = {}
        # for test no successor
        edges['c2p'] = [(1, 4), (0, 5), (1, 9), (1, 8), (2, 8), (2, 5), (3, 6),
                        (3, 7), (3, 4), (3, 8)]
        edges['p2c'] = [(v, u) for u, v in edges['c2p']]
        edges['p2a'] = [(4, 10), (4, 11), (4, 12), (4, 14), (4, 13), (6, 12),
                        (6, 11), (6, 14), (7, 12), (7, 11), (8, 14), (9, 10)]
        edges['a2p'] = [(v, u) for u, v in edges['p2a']]

        node_types = ['c' for _ in range(4)] + \
                     ['p' for _ in range(6)] + \
                     ['a' for _ in range(5)]
        node_types = [(i, t) for i, t in enumerate(node_types)]

        nfeat = {'nfeat': np.random.randn(num_nodes, dim)}
        efeat = {}
        for etype, _edges in edges.items():
            efeat[etype] = {'efeat': np.random.randn(len(_edges), dim)}

        hg = pgl.HeterGraph(edges=edges,
                node_types=node_types,
                node_feat=nfeat,
                edge_feat=efeat)

        # inplace
        new_hg = hg.tensor(inplace=False)
        self.assertNotIsInstance(hg.node_feat['nfeat'], paddle.Tensor)
        self.assertNotIsInstance(hg.edge_feat['a2p']['efeat'], paddle.Tensor)

        self.assertIsInstance(new_hg.node_feat['nfeat'], paddle.Tensor)
        self.assertIsInstance(new_hg.edge_feat['a2p']['efeat'], paddle.Tensor)
        self.assertIsInstance(new_hg.num_nodes, paddle.Tensor)

        hg.tensor(inplace=True)
        self.assertIsInstance(hg.node_feat['nfeat'], paddle.Tensor)
        self.assertIsInstance(hg.edge_feat['a2p']['efeat'], paddle.Tensor)

        new_hg = hg.numpy(inplace=False)
        self.assertIsInstance(new_hg.node_feat['nfeat'], np.ndarray)
        self.assertIsInstance(new_hg.edge_feat['a2p']['efeat'], np.ndarray)
        self.assertIsInstance(hg.node_feat['nfeat'], paddle.Tensor)
        self.assertIsInstance(hg.edge_feat['a2p']['efeat'], paddle.Tensor)

        hg.numpy(inplace=True)
        self.assertIsInstance(hg.node_feat['nfeat'], np.ndarray)
        self.assertIsInstance(hg.edge_feat['a2p']['efeat'], np.ndarray)

    def test_dump_and_load(self):
        np.random.seed(1)
        dim = 4
        num_nodes = 15

        edges = {}
        # for test no successor
        edges['c2p'] = [(1, 4), (0, 5), (1, 9), (1, 8), (2, 8), (2, 5), (3, 6),
                        (3, 7), (3, 4), (3, 8)]
        edges['p2c'] = [(v, u) for u, v in edges['c2p']]
        edges['p2a'] = [(4, 10), (4, 11), (4, 12), (4, 14), (4, 13), (6, 12),
                        (6, 11), (6, 14), (7, 12), (7, 11), (8, 14), (9, 10)]
        edges['a2p'] = [(v, u) for u, v in edges['p2a']]

        node_types = ['c' for _ in range(4)] + \
                     ['p' for _ in range(6)] + \
                     ['a' for _ in range(5)]
        node_types = [(i, t) for i, t in enumerate(node_types)]

        nfeat = {'nfeat': np.random.randn(num_nodes, dim)}
        efeat = {}
        for etype, _edges in edges.items():
            efeat[etype] = {'efeat': np.random.randn(len(_edges), dim)}

        hg = pgl.HeterGraph(edges=edges,
                node_types=node_types,
                node_feat=nfeat,
                edge_feat=efeat)

        path = "./tmp"
        hg.dump(path, indegree=True)

        hg2 = pgl.HeterGraph.load(path)

        self.assertEqual(hg.num_nodes, hg2.num_nodes)

        del hg
        del hg2
        shutil.rmtree(path)


if __name__ == "__main__":
    unittest.main()
