import os
import numpy as np
from tqdm import tqdm
from ogb.lsc import MAG240MDataset

if __name__ == '__main__':
    root = 'dataset_path'
    m2v_input_path = 'your_m2v_input_path'
    m2v_output_path = 'your_m2v_output_path'
    dataset = MAG240MDataset(root)
    N = (dataset.num_papers + dataset.num_authors + dataset.num_institutions)
    part_num = (N + 1) // 10
    start_idx = 0
    node_chunk_size = 100000
    m2v_merge_feat = np.memmap(m2v_output_path, dtype=np.float16, mode='w+', shape=(N, 64))
    files = os.listdir(m2v_input_path)
    files = sorted(files)
    for idx, start_idx in enumerate(range(0, N, part_num)):
        end_idx = min(N, start_idx + part_num)
        f = os.path.join(m2v_input_path, files[idx])
        m2v_feat_tmp = np.memmap(f, dtype=np.float16, mode='r', shape=(end_idx - start_idx, 64))
        for i in tqdm(range(start_idx, end_idx, node_chunk_size)):
            j = min(i + node_chunk_size, end_idx)
            m2v_merge_feat[i: j] = m2v_feat_tmp[i - start_idx: j - start_idx]
        m2v_merge_feat.flush()
        del m2v_feat_tmp
