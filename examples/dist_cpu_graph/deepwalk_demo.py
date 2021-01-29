from pgl.contrib.distributed.graph_client import DistCPUGraphClient
from pgl.contrib.distributed.dist_sample import metapath_randomwalk

g = DistCPUGraphClient("server_endpoints", shard_num=2)

batch_size = 128
n_type = "u"
print(g.sample_predecessor([1,3,2,10], 10, "u2t"))

for batch_node in g.node_batch_iter(batch_size,
                            node_type=n_type,
                            shuffle=True):
    print("batch_node", batch_node)
    walks = metapath_randomwalk(graph=g,
                                start_nodes=batch_node,
                                metapath="u2t-t2u",
                                walk_length=10)
    print("walks", walks)
