import os
import yaml
import pgl
import time
import copy
import numpy as np
import os.path as osp
from pgl.utils.logger import log
from pgl.bigraph import BiGraph
from pgl import graph_kernel
from pgl.sampling.custom import subgraph
from ogb.lsc import MAG240MDataset, MAG240MEvaluator
import time
import paddle
from tqdm import tqdm
from pgl.utils.helper import scatter


def get_col_slice(x, start_row_idx, end_row_idx, start_col_idx, end_col_idx):
    outs = []
    chunk = 100000
    for i in tqdm(range(start_row_idx, end_row_idx, chunk)):
        j = min(i + chunk, end_row_idx)
        outs.append(x[i:j, start_col_idx:end_col_idx].copy())
    return np.concatenate(outs, axis=0)


def save_col_slice(x_src, x_dst, start_row_idx, end_row_idx, start_col_idx,
                   end_col_idx):
    assert x_src.shape[0] == end_row_idx - start_row_idx
    assert x_src.shape[1] == end_col_idx - start_col_idx
    chunk, offset = 100000, start_row_idx
    for i in tqdm(range(0, end_row_idx - start_row_idx, chunk)):
        j = min(i + chunk, end_row_idx - start_row_idx)
        x_dst[offset + i:offset + j, start_col_idx:end_col_idx] = x_src[i:j]


class MAG240M(object):
    """Iterator"""
    def __init__(self, data_dir, seed=123):
        self.data_dir = data_dir
        self.num_features = 768
        self.num_classes = 153
        self.seed = seed
    
    def prepare_data(self):
        dataset = MAG240MDataset(self.data_dir)
        
        log.info(dataset.num_authors)
        log.info(dataset.num_institutions)
        
        author_path = f'{dataset.dir}/author_feat.npy'
        path = f'{dataset.dir}/institution_feat.npy'
        t = time.perf_counter()
        if not osp.exists(path):
            log.info('get institution_feat...')
            
            author_feat = np.memmap(author_path, dtype=np.float16,
                                    shape=(dataset.num_authors, self.num_features),
                                    mode='r')
            # author
            edge_index = dataset.edge_index('author', 'institution')
            edge_index = edge_index.T
            log.info(edge_index.shape)
            institution_graph = BiGraph(edge_index, dst_num_nodes=dataset.num_institutions)
            institution_graph.tensor()
            log.info('finish institution graph')
            
            institution_x = np.memmap(path, dtype=np.float16, mode='w+',
                          shape=(dataset.num_institutions, self.num_features))
            dim_chunk_size = 64
            
            degree = paddle.zeros(shape=[dataset.num_institutions, 1], dtype='float32')
            temp_one = paddle.ones(shape=[edge_index.shape[0], 1], dtype='float32')
            degree = scatter(degree, overwrite=False, index=institution_graph.edges[:, 1],
                            updates=temp_one)
            log.info('finish degree')
            
            for i in tqdm(range(0, self.num_features, dim_chunk_size)):
                j = min(i + dim_chunk_size, self.num_features)
                inputs = get_col_slice(author_feat, start_row_idx=0,
                                       end_row_idx=dataset.num_authors,
                                       start_col_idx=i, end_col_idx=j)
                
                inputs = paddle.to_tensor(inputs, dtype='float32')
                outputs = institution_graph.send_recv(inputs)
                outputs = outputs / degree
                outputs = outputs.astype('float16').numpy()
                
                del inputs
                save_col_slice(
                    x_src=outputs, x_dst=institution_x, start_row_idx=0,
                    end_row_idx=dataset.num_institutions,
                    start_col_idx=i, end_col_idx=j)
                del outputs
                
            institution_x.flush()
            del institution_x
            log.info(f'Done! [{time.perf_counter() - t:.2f}s]')
            
if __name__ == "__main__":
    root = 'dataset_path'
    print(root)
    dataset = MAG240M(root)
    dataset.prepare_data()
            
