import numpy as np
from ogb.lsc import MAG240MDataset
import os

def create_split_dir(source_dir, split_dir):
    dataset = MAG240MDataset(source_dir)
    split_dir_exists = os.path.exists(split_dir)
    if not split_dir_exists:
        os.mkdir(split_dir)

    valid_idx = dataset.get_idx_split("valid")
    np.random.seed(999)
    np.random.shuffle(valid_idx)
    end = len(valid_idx)
    part = len(valid_idx) // 5 + 1

    for idx, x in enumerate(range(0, end, part)):
        y = min(x + part, end)
        valid_part = valid_idx[x: y]
        print(valid_part.shape)
        split_file = f"{split_dir}/valid_{idx}"
        np.save(split_file, valid_part)


if __name__ == "__main__":
    source_data_dir = 'your_source_data_dir'
    split_data_dir = 'your_split_data_dir'
    create_split_dir(source_dir=source_data_dir, split_dir=split_data_dir)
