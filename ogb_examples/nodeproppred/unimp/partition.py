"""Loader"""
import numpy as np
import math
from pgl.sample import extract_edges_from_nodes

def random_partition(num_clusters, graph, shuffle=True):
    """random partition"""
    batch_size = int(math.ceil(graph.num_nodes / num_clusters))
    perm = np.arange(0, graph.num_nodes)
    if shuffle:
        np.random.shuffle(perm)

    batch_no = 0
 
    while batch_no < graph.num_nodes:
        batch_nodes = perm[batch_no:batch_no + batch_size]
        batch_no += batch_size
        eids = extract_edges_from_nodes(graph, batch_nodes)
        sub_g = graph.subgraph(nodes=batch_nodes, eid=eids,
                  with_node_feat=True, with_edge_feat=False)
        for key, value in graph.edge_feat.items():
            sub_g.edge_feat[key] = graph.edge_feat[key][eids]
        yield sub_g
        
        

def random_partition_v2(num_clusters, graph, shuffle=True, save_e=[]):
    """random partition v2"""
    if shuffle:
        cluster_id = np.random.randint(low=0, high=num_clusters, size=graph.num_nodes)
    else:
        if not save_e:
            cluster_id = np.random.randint(low=0, high=num_clusters, size=graph.num_nodes)
            save_e.append(cluster_id)
        else:
            cluster_id = save_e[0]
#         assert cluster_id is not None   
    perm = np.arange(0, graph.num_nodes)
    batch_no = 0
    while batch_no < num_clusters:
        batch_nodes = perm[cluster_id == batch_no]
        batch_no += 1 
        eids = extract_edges_from_nodes(graph, batch_nodes)
        sub_g = graph.subgraph(nodes=batch_nodes, eid=eids,
            with_node_feat=True, with_edge_feat=False)
        for key, value in graph.edge_feat.items():
            sub_g.edge_feat[key] = graph.edge_feat[key][eids]
        yield sub_g
