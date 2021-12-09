import os
import tqdm
import yaml
import pgl
import time
import copy
import numpy as np
import os.path as osp
from pgl.utils.logger import log
from pgl.graph import Graph
from pgl import graph_kernel
from pgl.sampling.custom import subgraph
from ogb.lsc import MAG240MDataset, MAG240MEvaluator
from dataset.base_dataset import BaseDataGenerator
import time


class MAG240M(object):
    """Iterator"""
    def __init__(self, data_dir, seed=123):
        self.data_dir = data_dir
        self.num_features = 768
        self.num_classes = 153
        self.seed = seed
    
    def prepare_data(self):
        dataset = MAG240MDataset(self.data_dir)
        edge_path = f'{dataset.dir}/paper_to_paper_symmetric_pgl'
        
        t = time.perf_counter()
        if not osp.exists(edge_path):
            log.info('Converting adjacency matrix...')
            edge_index = dataset.edge_index('paper', 'cites', 'paper')
            edge_index = edge_index.T
            
            edges_new = np.zeros((edge_index.shape[0], 2))
            edges_new[:, 0] = edge_index[:, 1]
            edges_new[:, 1] = edge_index[:, 0]
            
            edge_index = np.vstack((edge_index, edges_new))
            edge_index = np.unique(edge_index, axis=0)
            
            graph = Graph(edge_index, sorted=True)
            graph.adj_dst_index
            graph.dump(edge_path)
            log.info(f'Done! [{time.perf_counter() - t:.2f}s]')
        
        np.random.seed(self.seed)
        self.train_idx = dataset.get_idx_split('train')
#         np.random.shuffle(self.train_idx)
        
        self.val_idx = dataset.get_idx_split('valid')
        self.test_idx = dataset.get_idx_split('test')

        self.x = dataset.paper_feat
        self.y = dataset.all_paper_label

        self.graph = Graph.load(edge_path, mmap_mode='r+')
        log.info(f'Done! [{time.perf_counter() - t:.2f}s]')
    
    @property
    def train_examples(self,):
        # Filters
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        trainer_num = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
        count_line = 0
        
        np.random.shuffle(self.train_idx)
        for idx in self.train_idx:
            count_line += 1
            if count_line % trainer_num == trainer_id:
                yield idx
                
    @property        
    def eval_examples(self, ):
        for idx in self.val_idx:
            yield idx
                
    @property
    def test_examples(self,):
        for idx in self.test_idx:
            yield idx
                
                
def add_self_loop(graph, sub_nodes=None):
    '''add_self_loop_for_subgraph
    '''
    assert not graph.is_tensor(), "You must call Graph.numpy() first."
    
    if sub_nodes is not None:
        self_loop_edges = np.zeros((sub_nodes.shape[0], 2))
        self_loop_edges[:, 0] = self_loop_edges[:, 1] = sub_nodes
    else:
        self_loop_edges = np.zeros((graph.num_nodes, 2))
        self_loop_edges[:, 0] = self_loop_edges[:, 1] = np.arange(graph.num_nodes)
    edges = np.vstack((graph.edges, self_loop_edges))
    edges = np.unique(edges, axis=0)
    new_g = Graph(
        edges=edges,
        num_nodes=graph.num_nodes,
        node_feat=graph.node_feat,
        edge_feat=graph.edge_feat)
    return new_g


def traverse(item):
    """traverse
    """
    if isinstance(item, list) or isinstance(item, np.ndarray):
        for i in iter(item):
            for j in traverse(i):
                yield j
    else:
        yield item


def flat_node_and_edge(nodes):
    """flat_node_and_edge
    """
    nodes = list(set(traverse(nodes)))
    return nodes
                
def neighbor_sample(graph, nodes, samples):
    assert not graph.is_tensor(), "You must call Graph.numpy() first."
    
    graph_list = []
    for max_deg in samples:
        start_nodes = copy.deepcopy(nodes)
        edges = []
        if max_deg == -1:
            pred_nodes = graph.predecessor(start_nodes)
        else:
            pred_nodes = graph.sample_predecessor(start_nodes, max_degree=max_deg)

        for dst_node, src_nodes in zip(start_nodes, pred_nodes):
            for src_node in src_nodes:
                edges.append((src_node, dst_node))
        
        neigh_nodes = [start_nodes, pred_nodes]
        neigh_nodes = flat_node_and_edge(neigh_nodes)
        
        
        from_reindex = {x: i for i, x in enumerate(neigh_nodes)}
        sub_node_index = graph_kernel.map_nodes(nodes, from_reindex)
        
        sg = subgraph(graph,
                      nodes=neigh_nodes,
                      edges=edges,
                      with_node_feat=False,
                      with_edge_feat=False)
        
        sg = add_self_loop(sg, sub_node_index)
        
        graph_list.append((sg, neigh_nodes, sub_node_index))
        nodes = neigh_nodes
        
    graph_list = graph_list[::-1] 
    return graph_list
                
        
class DataGenerator(BaseDataGenerator):
    def __init__(self, dataset, samples, batch_size, num_workers, data_type):

        super(DataGenerator, self).__init__(buf_size=1000, 
                                            batch_size=batch_size, 
                                            num_workers=num_workers, 
                                            shuffle=True if data_type=='train' else False)
        
        self.dataset = dataset
        self.samples = samples
        if data_type == 'train':
            self.line_examples = self.dataset.train_examples
        elif data_type == 'eval':
            self.line_examples = self.dataset.eval_examples
        else: 
            self.line_examples = self.dataset.test_examples

    def batch_fn(self, batch_nodes):
        
        graph_list = neighbor_sample(self.dataset.graph, batch_nodes, 
                                     self.samples)
        
        neigh_nodes = graph_list[0][1]
        
#         x = self.dataset.x[neigh_nodes]
        
        y = self.dataset.y[batch_nodes]
        return graph_list, neigh_nodes, y
