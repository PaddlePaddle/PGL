import os
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
import time
from tqdm import tqdm


class MAG240M(object):
    """Iterator"""
    def __init__(self, data_dir):
        self.num_features = 768
        self.num_classes = 153
        self.data_dir = data_dir
    
    def prepare_data(self):
        dataset = MAG240MDataset(self.data_dir)
        
        graph_file_list = []
        paper_edge_path = f'{dataset.dir}/paper_to_paper_symmetric_pgl_split'
        graph_file_list.append(paper_edge_path)
        t = time.perf_counter()
        if not osp.exists(paper_edge_path):
            log.info('Converting adjacency matrix...')
            edge_index = dataset.edge_index('paper', 'cites', 'paper')
            edge_index = edge_index.T
            
            edges_new = np.zeros((edge_index.shape[0], 2))
            edges_new[:, 0] = edge_index[:, 1]
            edges_new[:, 1] = edge_index[:, 0]
            edge_index = np.vstack((edge_index, edges_new))
            edge_types = np.full([edge_index.shape[0], ], 0, dtype='int32')
            
            graph = Graph(edge_index, num_nodes=dataset.num_papers, edge_feat={'edge_type': edge_types})
            graph.adj_dst_index
            graph.dump(paper_edge_path)
            log.info(f'Done! [{time.perf_counter() - t:.2f}s]')
        
        author_edge_path = f'{dataset.dir}/paper_to_author_symmetric_pgl_split_src'
        graph_file_list.append(author_edge_path)
        t = time.perf_counter()
        if not osp.exists(author_edge_path):
            log.info('Converting author matrix...')
            
            # author
            log.info('adding author edges')
            edge_index = dataset.edge_index('author', 'writes', 'paper')
            edge_index = edge_index.T
            row, col = edge_index[:, 0], edge_index[:, 1]
            log.info(row[:10])
            row += dataset.num_papers
            
            edge_types = np.full(row.shape, 1, dtype='int32')
            edge_index = np.stack([row, col], axis=1)
            
            graph = Graph(edge_index, edge_feat={'edge_type': edge_types})
            graph.adj_dst_index
            graph.dump(author_edge_path)
            log.info(f'Done! finish author_edge [{time.perf_counter() - t:.2f}s]')
            
        
        author_edge_path = f'{dataset.dir}/paper_to_author_symmetric_pgl_split_dst'
        graph_file_list.append(author_edge_path)
        t = time.perf_counter()
        if not osp.exists(author_edge_path):
            log.info('Converting author matrix...')
            
            # author
            log.info('adding author edges')
            edge_index = dataset.edge_index('author', 'writes', 'paper')
            edge_index = edge_index.T
            row, col = edge_index[:, 0], edge_index[:, 1]
            log.info(row[:10])
            row += dataset.num_papers
            
            edge_types = np.full(row.shape, 2, dtype='int32')
            edge_index = np.stack([col, row], axis=1)
            
            graph = Graph(edge_index, edge_feat={'edge_type': edge_types})
            graph.adj_dst_index
            graph.dump(author_edge_path)
            log.info(f'Done! finish author_edge [{time.perf_counter() - t:.2f}s]')
        
        
        institution_edge_path = f'{dataset.dir}/institution_edge_symmetric_pgl_split_src'
        graph_file_list.append(institution_edge_path)
        t = time.perf_counter()
        if not osp.exists(institution_edge_path):
            log.info('Converting institution matrix...')
            
            # institution
            log.info('adding institution edges')
            edge_index = dataset.edge_index('author', 'institution')
            edge_index = edge_index.T
            row, col = edge_index[:, 0], edge_index[:, 1]
            log.info(row[:10])
            row += dataset.num_papers
            col += dataset.num_papers + dataset.num_authors
            
            # edge_type
            log.info('building edge type')
            edge_types = np.full(row.shape, 3, dtype='int32')
            edge_index = np.stack([row, col], axis=1)
            
            graph = Graph(edge_index, edge_feat={'edge_type': edge_types})
            graph.adj_dst_index
            graph.dump(institution_edge_path)
            log.info(f'Done! finish institution_edge [{time.perf_counter() - t:.2f}s]')
            
            
        institution_edge_path = f'{dataset.dir}/institution_edge_symmetric_pgl_split_dst'
        graph_file_list.append(institution_edge_path)
        t = time.perf_counter()
        if not osp.exists(institution_edge_path):
            log.info('Converting institution matrix...')
            
            # institution
            log.info('adding institution edges')
            edge_index = dataset.edge_index('author', 'institution')
            edge_index = edge_index.T
            row, col = edge_index[:, 0], edge_index[:, 1]
            log.info(row[:10])
            row += dataset.num_papers
            col += dataset.num_papers + dataset.num_authors
            
            # edge_type
            log.info('building edge type')
            edge_types = np.full(row.shape, 4, dtype='int32')
            edge_index = np.stack([col, row], axis=1)
            
            graph = Graph(edge_index, edge_feat={'edge_type': edge_types})
            graph.adj_dst_index
            graph.dump(institution_edge_path)
            log.info(f'Done! finish institution_edge [{time.perf_counter() - t:.2f}s]')

            
        path = f'{dataset.dir}/full_feat.npy'
        
        author_feat_path = f'{dataset.dir}/author_feat.npy'
        
        institution_feat_path = f'{dataset.dir}/institution_feat.npy'
        
        t = time.perf_counter()
        if not osp.exists(path):  # Will take ~3 hours...
            print('Generating full feature matrix...')

            node_chunk_size = 100000
            N = (dataset.num_papers + dataset.num_authors +
                 dataset.num_institutions)

            paper_feat = dataset.paper_feat
            
            author_feat = np.memmap(author_feat_path, dtype=np.float16,
                                    shape=(dataset.num_authors, self.num_features),
                                    mode='r')
            
            institution_feat = np.memmap(institution_feat_path, dtype=np.float16,
                                    shape=(dataset.num_institutions, self.num_features),
                                    mode='r')
            
            x = np.memmap(path, dtype=np.float16, mode='w+',
                          shape=(N, self.num_features))
            
            print('Copying paper features...')
            start_idx = 0 
            end_idx = dataset.num_papers
            for i in tqdm(range(start_idx, end_idx, node_chunk_size)):
                j = min(i + node_chunk_size, end_idx)
                x[i: j] = paper_feat[i: j]
            del paper_feat
            
            print('Copying author feature...')
            start_idx = dataset.num_papers
            end_idx = dataset.num_papers + dataset.num_authors
            for i in tqdm(range(start_idx, end_idx, node_chunk_size)):
                j = min(i + node_chunk_size, end_idx)
                x[i: j] = author_feat[i - start_idx: j - start_idx]
            del author_feat
            
            print('Copying institution feature...')
            start_idx = dataset.num_papers + dataset.num_authors
            end_idx = dataset.num_papers + dataset.num_authors + dataset.num_institutions
            for i in tqdm(range(start_idx, end_idx, node_chunk_size)):
                j = min(i + node_chunk_size, end_idx)
                x[i: j] = institution_feat[i - start_idx: j - start_idx]
            del institution_feat
            
            x.flush()
            del x
            print(f'feature x Done! [{time.perf_counter() - t:.2f}s]')
        
        
        path = f'{dataset.dir}/all_feat_year.npy'
        
        author_year_path = f'{dataset.dir}/author_feat_year.npy'
        
        institution_year_path = f'{dataset.dir}/institution_feat_year.npy'
        
        t = time.perf_counter()
        if not osp.exists(path):  # Will take ~3 hours...
            print('Generating full year matrix...')

            node_chunk_size = 100000
            N = (dataset.num_papers + dataset.num_authors +
                 dataset.num_institutions)

            paper_year_feat = dataset.all_paper_year
            
            author_year_feat = np.memmap(author_year_path, dtype=np.int32,
                                    shape=(dataset.num_authors),
                                    mode='r')
            
            institution_year_feat = np.memmap(institution_year_path, dtype=np.int32,
                                    shape=(dataset.num_institutions),
                                    mode='r')
            
            x = np.memmap(path, dtype=np.int32, mode='w+',
                          shape=(N))
            
            print('Copying paper features...')
            start_idx = 0 
            end_idx = dataset.num_papers
            for i in tqdm(range(start_idx, end_idx, node_chunk_size)):
                j = min(i + node_chunk_size, end_idx)
                x[i: j] = paper_year_feat[i: j]
            del paper_year_feat
            
            print('Copying author feature...')
            start_idx = dataset.num_papers
            end_idx = dataset.num_papers + dataset.num_authors
            for i in tqdm(range(start_idx, end_idx, node_chunk_size)):
                j = min(i + node_chunk_size, end_idx)
                x[i: j] = author_year_feat[i - start_idx: j - start_idx]
            del author_year_feat
            
            print('Copying institution feature...')
            start_idx = dataset.num_papers + dataset.num_authors
            end_idx = dataset.num_papers + dataset.num_authors + dataset.num_institutions
            for i in tqdm(range(start_idx, end_idx, node_chunk_size)):
                j = min(i + node_chunk_size, end_idx)
                x[i: j] = institution_year_feat[i - start_idx: j - start_idx]
            del institution_year_feat
            
            x.flush()
            del x
            print(f'year feature Done! [{time.perf_counter() - t:.2f}s]')
                
if __name__ == "__main__":
    root = 'dataset_path'
    print(root)
    dataset = MAG240M(root)
    dataset.prepare_data()


