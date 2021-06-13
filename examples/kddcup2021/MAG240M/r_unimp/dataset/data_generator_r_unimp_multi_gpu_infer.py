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
from dataset.base_dataset import BaseDataGenerator
import time
from tqdm import tqdm


class MAG240M(object):
    """Iterator"""
    def __init__(self, config):
        self.num_features = 768
        self.num_classes = 153
        self.data_dir = config.data_dir
        self.seed = config.seed
        self.valid_path = config.valid_path
        self.valid_name = config.valid_name
        self.m2v_file = config.m2v_file
        self.m2v_dim = config.m2v_dim

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
            print(f'Done! [{time.perf_counter() - t:.2f}s]')

        np.random.seed(self.seed)
        self.train_idx = dataset.get_idx_split('train')
        self.val_idx = dataset.get_idx_split('valid')
        valid_name = os.path.join(self.valid_path, self.valid_name)
        self.val_idx_cv = np.load(valid_name)
        log.info(self.train_idx.shape)
        log.info(self.val_idx.shape)
        log.info(self.val_idx_cv.shape)
        self.test_idx = dataset.get_idx_split('test')
        ##self.val_idx = np.load('valid_idx_eval.npy')
        def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):

            def cal_angle(position, hid_idx):
                return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)
            def get_posi_angle_vec(position):
                return [cal_angle(position, hid_j) for hid_j in range(d_hid)]
            sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
            sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
            sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
            return sinusoid_table

        N = dataset.num_papers + dataset.num_authors + dataset.num_institutions
        self.x = np.memmap(f'{dataset.dir}/full_feat.npy', dtype=np.float16,
                           mode='r', shape=(N, self.num_features))

        self.id_x = np.memmap(f'{dataset.dir}/{self.m2v_file}', dtype=np.float16,
                          mode='r', shape=(N, self.m2v_dim))
        
        self.y = dataset.all_paper_label

        self.graph = [Graph.load(edge_path, mmap_mode='r+') for edge_path in graph_file_list]

        self.pos = get_sinusoid_encoding_table(200, 768)
        #self.year = dataset.all_paper_year
        year_file = f'{dataset.dir}/all_feat_year.npy'
        self.year = np.memmap(year_file, dtype=np.int32, mode='r',
              shape=(N,))
        self.num_papers = dataset.num_papers
        self.train_idx_label = None
        self.train_idx_data = None
        log.info(f'Done! [{time.perf_counter() - t:.2f}s]')

    @property
    def train_examples(self,):
        # Filters
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        trainer_num = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
        count_line = 0

        np.random.shuffle(self.train_idx)
        if self.train_idx_label is not None:
            del self.train_idx_label
        if self.train_idx_data is not None:
            del self.train_idx_data
        #self.train_idx_label = set(self.train_idx[: len(self.train_idx) // 2])
        #self.train_idx_data = self.train_idx[len(self.train_idx) // 2: ]
        self.train_idx_label = set(self.train_idx)
        self.train_idx_data = self.train_idx

        for idx in self.train_idx_data:
            count_line += 1
            if count_line % trainer_num == trainer_id:
                yield idx

    @property
    def eval_examples(self, ):
        if self.train_idx_label is not None:
            del self.train_idx_label
        if self.train_idx_data is not None:
            del self.train_idx_data
        #self.train_idx_label = set(self.train_idx) | set(np.load("valid_idx_train.npy"))
        self.train_idx_label = set(self.train_idx) | set(self.val_idx)

        trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        trainer_num = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
        count_line = 0
        log.info("finish label_idx update in valid")
        for idx in self.val_idx_cv:
            count_line += 1
            if count_line % trainer_num == trainer_id:
                yield idx

    @property
    def test_examples(self,):
        if self.train_idx_label is not None:
            del self.train_idx_label
        if self.train_idx_data is not None:
            del self.train_idx_data
            
        self.train_idx_label = set(self.train_idx) | set(self.val_idx)
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        trainer_num = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
        count_line = 0
        log.info("finish label_idx update in test")
        for idx in self.test_idx:
            count_line += 1
            if count_line % trainer_num == trainer_id:
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
        )
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
    graph_list = []
    for max_deg in samples:
        start_nodes = copy.deepcopy(nodes)
        edges = []
        edge_ids = []
        edge_feats = []
        neigh_nodes = [start_nodes]
        if max_deg == -1:
            pred_nodes, pred_eids = graph.predecessor(start_nodes, return_eids=True)
        else:
            for g_t in graph:
                pred_nodes, pred_eids = g_t.sample_predecessor(start_nodes, max_degree=max_deg, return_eids=True)
                neigh_nodes.append(pred_nodes)
                for dst_node, src_nodes, src_eids in zip(start_nodes, pred_nodes, pred_eids):
                    for src_node, src_eid in zip(src_nodes, src_eids):
                        edges.append((src_node, dst_node))
                        edge_ids.append(src_eid)
                        edge_feats.append(g_t.edge_feat['edge_type'][src_eid])
        neigh_nodes = flat_node_and_edge(neigh_nodes)

        from_reindex = {x: i for i, x in enumerate(neigh_nodes)}
        sub_node_index = graph_kernel.map_nodes(nodes, from_reindex)

        sg = subgraph(graph[0],
                      eid=edge_ids,
                      nodes=neigh_nodes,
                      edges=edges,
                      with_node_feat=False,
                      with_edge_feat=False)
        edge_feats = np.array(edge_feats, dtype='int32')
        sg._edge_feat['edge_type'] = edge_feats

        graph_list.append((sg, neigh_nodes, sub_node_index))
        nodes = neigh_nodes

    graph_list = graph_list[::-1]
    return graph_list, from_reindex


class DataGenerator(BaseDataGenerator):
    def __init__(self, dataset, samples, batch_size, num_workers, data_type):

        super(DataGenerator, self).__init__(buf_size=10,
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
        self.data_type = data_type
        
        
    def batch_fn(self, batch_nodes):

        graph_list, from_reindex = neighbor_sample(self.dataset.graph, batch_nodes,
                                     self.samples)

        neigh_nodes = graph_list[0][1]
        neigh_nodes = np.array(neigh_nodes, dtype='int32')
        y = self.dataset.y[batch_nodes]
        if self.data_type == "test":
            label_idx = list(set(neigh_nodes) & self.dataset.train_idx_label)
        else:
            label_idx = list((set(neigh_nodes) - set(batch_nodes)) & self.dataset.train_idx_label)
        sub_label_index = graph_kernel.map_nodes(label_idx, from_reindex)
        sub_label_y = self.dataset.y[label_idx]
        pos = 2021 - self.dataset.year[neigh_nodes]
        return graph_list, neigh_nodes, y, sub_label_y, sub_label_index, pos, batch_nodes

    def post_fn(self, batch):
        graph_list, neigh_nodes, y, sub_label_y, sub_label_index, pos, batch_nodes = batch
        x = self.dataset.x[neigh_nodes]
        id_x = self.dataset.id_x[neigh_nodes]
        pos = self.dataset.pos[pos]
        x = x + pos
        return graph_list, x, id_x, y, sub_label_y, sub_label_index, batch_nodes

if __name__ == "__main__":
    root = 'dataset_path'
    print(root)
    dataset = MAG240M(root)
    dataset.prepare_data()
