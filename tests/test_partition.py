import os

import unittest
import numpy as np
import paddle
import pgl
from pgl.partition import metis_partition

from testsuite import create_random_graph


class GraphPartitionTest(unittest.TestCase):
    """ Test METIS Partition
    """
    def test_partition_graph(self):
        glist = []
        npart = 20 
        for i in range(npart):
            num_nodes = 10
            edges = np.random.randint(
                low=1,
                high=num_nodes,
                size=[np.random.randint(
                    low=1, high=10), 2])

            g = pgl.Graph(
                edges=edges,
                num_nodes=num_nodes)
            glist.append(g)

        # Merge Graph
        multi_graph = pgl.Graph.batch(glist)
        # Check Graph Index
        cluster_id = metis_partition(multi_graph, npart=npart)
        ori_graph_id = multi_graph.graph_node_id
        for i in range(npart):
            cluster = cluster_id[ori_graph_id  == i]
            # The partition should be in same within a cluster
            self.assertTrue(np.any(cluster != cluster[0]))
        

if __name__ == "__main__":
    unittest.main()
