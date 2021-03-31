#-*- coding: utf-8 -*-
import os
import sys
import time
import tqdm
import argparse
import unittest
import shutil
import numpy as np

from paddle.fluid.core import GraphPyService, GraphPyServer, GraphPyClient
from pgl.utils.logger import log

from dist_graph import DistGraphClient, DistGraphServer

# configuration file
config = """
etype2files: "u2e2t:./tmp_distgraph_test/edges.txt"
symmetry: True

ntype2files: "u:./tmp_distgraph_test/node_types.txt,t:./tmp_distgraph_test/node_types.txt"
nfeat_info: [["u", "a", "float32", 1], ["u", "b", "int32", 2], ["u", "c", "string", 1], ["t", "a", "float32", 1], ["t", "b", "int32", 2]]

meta_path: "u2e2t-t2e2u;t2e2u-u2e2t"
first_node_type: "u;t"
node_batch_stream_shuffle_size: 10000

"""

edges_file = """37	45	0.34
37	145	0.31
37	112	0.21
96	48	1.4
96	247	0.31
96	111	1.21
59	45	0.34
59	145	0.31
59	122	0.21
97	48	0.34
98	247	0.31
7	222	0.91
7	234	0.09
37	333	0.21
47	211	0.21
47	113	0.21
47	191	0.21
34	131	0.21
34	121	0.21
39	131	0.21"""

node_file = """u	98	a 0.21	b 13 14	c hello1
u	97	a 0.22	b 13 14	c hello2
u	96	a 0.23	b 13 14	c hello3
u	7	a 0.24	b 13 14	c hello4
u	59	a 0.25	b 13 14	c hello5
t	48	a 0.91	b 213 14
u	47	a 0.21	b 13 14	c hello6
t	45	a 0.21	b 213 14
u	39	a 0.21	b 13 14	c hello7
u	37	a 0.21	b 13 14	c hello8
u	34	a 0.21	b 13 14	c hello9
t	333	a 0.21	b 213 14
t	247	a 0.21	b 213 14
t	234	a 0.21	b 213 14
t	222	a 0.21	b 213 14
t	211	a 0.21	b 213 14
t	191	a 0.21	b 213 14
t	145	a 0.11	b 213 14
t	131	a 0.31	b 213 14
t	122	a 0.41	b 213 14
t	121	a 0.51	b 213 14
t	113	a 0.61	b 213 14
t	112	a 0.71	b 213 14
t	111	a 0.81	b 213 14"""

tmp_path = "./tmp_distgraph_test"
if not os.path.exists(tmp_path):
    os.makedirs(tmp_path)

with open(os.path.join(tmp_path, "config.yaml"), 'w') as f:
    f.write(config)

with open(os.path.join(tmp_path, "edges.txt"), 'w') as f:
    f.write(edges_file)

with open(os.path.join(tmp_path, "node_types.txt"), 'w') as f:
    f.write(node_file)


def get_server_ip_addr():
    ip_addr_list = ["127.0.0.1:8243", "127.0.0.1:8242"]
    return ip_addr_list


class DistGraphTest(unittest.TestCase):
    """DistGraphTest
    """

    @classmethod
    def setUpClass(cls):
        config = os.path.join(tmp_path, "config.yaml")

        ip_addr = get_server_ip_addr()
        shard_num = 100
        cls.s1 = DistGraphServer(config, shard_num, ip_addr, server_id=0)
        cls.s2 = DistGraphServer(config, shard_num, ip_addr, server_id=1)

        cls.c1 = DistGraphClient(
            config,
            shard_num=shard_num,
            server_num=2,
            ip_config=ip_addr,
            client_id=0)

        cls.c2 = DistGraphClient(
            config,
            shard_num=shard_num,
            server_num=2,
            ip_config=ip_addr,
            client_id=1)

        cls.c1.load_edges()
        cls.c1.load_node_types()

    def test_random_sample_nodes(self):
        g_u_nodes = [98, 97, 96, 7, 59, 47, 39, 37, 34]

        g_t_nodes = [
            211, 333, 222, 121, 234, 48, 113, 145, 111, 247, 45, 112, 122, 131,
            191
        ]

        node_type = "u"
        size = 0
        nodes = self.c1.random_sample_nodes(node_type, size)
        self.assertEqual(nodes, [])

        node_type = "u"
        size = 100
        nodes = self.c1.random_sample_nodes(node_type, size)
        self.assertEqual(set(nodes) | set(g_u_nodes), set(g_u_nodes))
        self.assertEqual(len(set(nodes)), len(nodes))

        node_type = "t"
        size = 3
        nodes = self.c1.random_sample_nodes(node_type, size)
        self.assertEqual(set(nodes) | set(g_t_nodes), set(g_t_nodes))
        self.assertEqual(len(set(nodes)), len(nodes))

    def test_node_batch_iter(self):

        g_u_nodes = [98, 97, 96, 7, 59, 47, 39, 37, 34]

        g_t_nodes = [
            211, 333, 222, 121, 234, 48, 113, 145, 111, 247, 45, 112, 122, 131,
            191
        ]

        # test shuffle == True
        node_generator = self.c1.node_batch_iter(
            batch_size=2, node_type="t", shuffle=True, rank=0, nrank=1)
        count = 0
        nodes = []
        for idx, batch_nodes in enumerate(node_generator):
            nodes.extend(batch_nodes)
        self.assertEqual(len(nodes), len(g_t_nodes))
        self.assertEqual(set(nodes), set(g_t_nodes))

        # test shuffle == False
        node_generator = self.c1.node_batch_iter(
            batch_size=2, node_type="t", shuffle=False, rank=0, nrank=1)
        count = 0
        nodes = []
        for idx, batch_nodes in enumerate(node_generator):
            nodes.extend(batch_nodes)
        self.assertEqual(len(nodes), len(g_t_nodes))
        self.assertEqual(set(nodes), set(g_t_nodes))

        # Test multi clients node batch iter
        node_generator1 = self.c1.node_batch_iter(
            batch_size=2, node_type="u", shuffle=True, rank=0, nrank=2)

        node_generator2 = self.c2.node_batch_iter(
            batch_size=2, node_type="u", shuffle=True, rank=1, nrank=2)
        nodes1 = []
        for idx, batch_nodes in enumerate(node_generator1):
            nodes1.extend(batch_nodes)

        nodes2 = []
        for idx, batch_nodes in enumerate(node_generator2):
            nodes2.extend(batch_nodes)

        self.assertTrue((set(nodes1) & set(nodes2)) == set())
        all_nodes = set(nodes1) | set(nodes2)
        self.assertEqual(all_nodes, set(g_u_nodes))

    def test_node_feat(self):
        nfeat = self.c1.get_node_feat(
            nodes=[98, 7], node_type="u", feat_names="a")

        g_nfeat = [0.21, 0.24]
        for n, g in zip(nfeat[0], g_nfeat):
            self.assertAlmostEqual(n, g)

        nodes = [98, 7, 333, 247]
        feat_names = ["a", "b"]
        u_nfeat = self.c1.get_node_feat(
            nodes=nodes, node_type="u", feat_names=feat_names)

        t_nfeat = self.c1.get_node_feat(
            nodes=nodes, node_type="t", feat_names=feat_names)
        #  print(nodes)
        #  print(u_nfeat)
        #  print(t_nfeat)

        res = []
        for nfeat1, nfeat2 in zip(u_nfeat, t_nfeat):
            nfeat_list = []
            for feat1, feat2 in zip(nfeat1, nfeat2):
                if len(feat1) > 0:
                    nfeat_list.append(feat1)
                elif len(feat2) > 0:
                    nfeat_list.append(feat2)

            res.append(nfeat_list)

        #  print(res)

    def test_sample_successor(self):
        nodes = [98, 7]
        neighs = self.c1.sample_successor(nodes, 10, "u2e2t")

        g_neighs = [[247], [222, 234]]
        for a, b in zip(neighs, g_neighs):
            self.assertEqual(set(a), set(b))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(tmp_path)


if __name__ == "__main__":
    unittest.main()
