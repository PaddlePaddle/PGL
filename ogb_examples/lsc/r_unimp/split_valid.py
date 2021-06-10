import numpy as np
from ogb.lsc import MAG240MDataset, MAG240MEvaluator

def split_dir(data_dir, output_dit):
    dataset = MAG240MDataset(data_dir)
    valid_idx = dataset.get_idx_split("valid")
    np.random.seed(999)
    np.random.shuffle(valid_idx)
    star = 0
    end = len(valid_idx)
    part = len(valid_idx)//5 + 1
    
    for idx, x in enumerate(range(0, end, part)):
        y = min(x+part, end)
        valid_part = valid_idx[x: y]
        print(valid_part.shape)
        path_p = f"{output_dit}/valid_{idx}"
        np.save(path_p, valid_part)


if __name__ == "__main__":
    data_dir = ""
    output_dir = ''
    split_valid(data_dir, output_dir)