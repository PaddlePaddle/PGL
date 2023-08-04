import os
import numpy as np
import paddle
from paddle.framework import core

class DistFeat(object):
    """ Shard the feature into multi-gpu with NCCL and UVA to support large scale feature.

    Args:
        x: numpy array feature
        mode: use 'uva' or 'gpu' for feature storage.

    Usage:

        import paddle

        paddle.distributed.init_parallel_env()

        x = ...

        x = DistFeat(x, mode="gpu")

        batch = np.random.randint(0, 100, shape=1000)
 
        x = x[batch]

    """
    def __init__(self, x, mode="uva"):
        self.rank = paddle.distributed.get_rank() 
        self.nrank = paddle.distributed.get_world_size()
        self.part = x.shape[1] // self.nrank
        self.mode = mode

        if mode == "cpu":
            self.x = x
        elif mode == "uva":
            x = x[:, self.rank * self.part: (self.rank + 1) * self.part]
            x = self.to_contiguousarray(x)
            self.x = core.eager.to_uva_tensor(x, 0)
        elif mode  == "gpu":
            x = x[:, self.rank * self.part: (self.rank + 1) * self.part]
            self.x = paddle.to_tensor(x, dtype=x.dtype)

    @paddle.no_grad()
    def get_max_key_size(self, key_size):
        tensor_list = []
        paddle.distributed.all_gather(tensor_list, paddle.to_tensor([key_size]))
        return paddle.max(paddle.concat(tensor_list))
    
    @paddle.no_grad()
    def padding(self, key, pad_size):
        pad_key = paddle.zeros([pad_size], dtype="int64")
        return paddle.concat([key, pad_key])
   
    @paddle.no_grad()
    def __getitem__(self, key):
        if self.mode != "cpu":
            key_size = paddle.shape(key)[0]
            max_key_size = self.get_max_key_size(key_size)  
            batch = self.padding(key, max_key_size - key_size)
            idx = self.allgather(batch)
            tensor = self.x[idx]
            tensor = self.alltoall(tensor)
            return tensor[:key_size]
        else:
            x = self.x[key]
            return paddle.to_tensor(x, dtype=x.dtype)

    def to_contiguousarray(self, x):
        if x.data.c_contiguous is False:
            x = np.ascontiguousarray(x)
        return x
 
    def allgather(self, tensor):
        group = paddle.distributed.collective._get_default_group()
        tensor_shape = list(tensor.shape)
        tensor_shape[0] *= group.nranks
        out = paddle.empty(tensor_shape, tensor.dtype)
        task = group.process_group.all_gather(tensor, out)
        task.wait()
        return out

    def alltoall(self, in_tensor_list):
        group = paddle.distributed.collective._get_default_group()
        nrank = group.nranks
        tensor_shape = in_tensor_list.shape
        out = paddle.empty(tensor_shape, in_tensor_list.dtype)
        task = group.process_group.alltoall(in_tensor_list, out)
        task.wait()
        n = out.shape[0] // nrank
        dim = out.shape[1]
        out = out.reshape([nrank, n, dim]).transpose([1, 0, 2]).reshape([n, dim * nrank])
        return out
     

if __name__ == "__main__":
    import tqdm
    from ogb.lsc import MAG240MDataset

    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()
    dataset = MAG240MDataset("ogb2022")
    N = (dataset.num_papers + dataset.num_authors + dataset.num_institutions)

    dir = "./ogb2022/"

    x = np.memmap(dir + "/m2v_64.npy", dtype=np.float16,
        mode='r', shape=(N, 64))

    x = DistFeat(x, mode="uva")
    for i in tqdm.tqdm(range(100000)):
        batch = paddle.randint(0, N, shape=[12800], dtype="int64")
        emb = x[batch]
        print(emb.shape)
