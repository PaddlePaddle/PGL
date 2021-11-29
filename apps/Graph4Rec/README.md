# Graph4Rec: A Universal and Large-scale Toolkit with Graph Neural Networks for Recommender Systems

**Graph4Rec** is a universal and large-scale toolkit with graph neural networks for recommender systems.

## Requirements

Please install paddlepaddle and pgl before using Graph4Rec.

```
python -m pip install paddlepaddle==0.0.0 -f https://www.paddlepaddle.org.cn/whl/linux/cpu-mkl/develop.html

pip install pgl -U
```

## Usage

### Input format


Suppose there are three different types of nodes, namely "u", "t" and "f".
They can form three kinds of edges, namely "u2t (u-t)", "u2f (u-f)" and "t2f (t-f)".
Then the format of edges file is as follows:

```
src_node_id \t dst_node_id

for example:
12345 \t 23345
12345 \t 23346
12346 \t 23345
```

If there is an edge weight, one can directly add the weight behind each edge, the format is as follows:

```
src_node_id \t dst_node_id \t edge_weight

for example:
12345 \t 23345 \t 0.4
12345 \t 23346 \t 0.2
12346 \t 23345 \t 0.8
```

The format of the node types file is as follows:

```
node_type \t node_id

for example:

u \t 12345
u \t 12346
t \t 23345
f \t 23346
```

You can also find out the specific data format in `./toy_data`.


### PGL graph engine launching

Now we support distributed loading graph data using **PGL Graph Engine**. We also develop a simple tutorial to show how to launch a graph engine, please refer to [here](../../tutorials/working_with_distributed_graph_engine.ipynb).

To launch a distributed graph service, please follow the steps below.

#### IP address setting

The first step is to set the IP list for each graph server. Each IP address with port represents a server. In `./toy_data/ip_list.txt` file, we set up 4 graph server as follow for demo:

```
127.0.0.1:8813
127.0.0.1:8814
127.0.0.1:8815
127.0.0.1:8816
```

#### Launching graph engine by openmpi

Before launching the graph engine, you should set up the below hyper-parameters in configuration file in `./user_configs/metapath2vec.yaml`:

```
etype2files: "u2buy2t:./toy_data/u2buy2i.txt,u2click2i:./toy_data/u2click2i.txt"
ntype2files: "u:./toy_data/node_types.txt,i:./toy_data/node_types.txt"
symmetry: True
shard_num: 1000
```

Then, we can launch the graph engine with the help of OpenMPI.

```
mpirun -np 4 python -m pgl.distributed.launch --ip_config ./toy_data/ip_list.txt --conf ./user_configs/metapath2vec.yaml --mode mpi --shard_num 1000
```

#### Launching graph engine manually

If you didn't install OpenMPI, you can launch the graph engine manually. 

Fox example, if we want to use 4 servers, we should run the following command separately on 4 terminals.

```
# terminal 3
python -m pgl.distributed.launch --ip_config ./toy_data/ip_list.txt --conf ./user_configs/metapath2vec.yaml --shard_num 1000 --server_id 3

# terminal 2
python -m pgl.distributed.launch --ip_config ./toy_data/ip_list.txt --conf ./user_configs/metapath2vec.yaml --shard_num 1000 --server_id 2

# terminal 1
python -m pgl.distributed.launch --ip_config ./toy_data/ip_list.txt --conf ./user_configs/metapath2vec.yaml --shard_num 1000 --server_id 1

# terminal 0
python -m pgl.distributed.launch --ip_config ./toy_data/ip_list.txt --conf ./user_configs/metapath2vec.yaml --shard_num 1000 --server_id 0
```

Note that the `shard_num` should be **the same** as in configuration file.

After successfully launching the graph engine, you can run the below command to train the model in different modes.

### Single GPU training and inference

```
# Training
cd ./env_run/src
export CUDA_VISIBLE_DEVICES=0
python train.py --config ../../user_configs/metapath2vec.yaml --ip ../../toy_data/ip_list.txt

# Inference
python infer.py --config ../../user_configs/metapath2vec.yaml \
                --ip ../../toy_data/ip_list.txt \
                --save_dir ./save_embed/ \
                --infer_from ../../ckpt_custom/metapath2vec.0712/ckpt.pdparams

# The node embeddings will be saved in ./save_embed/embedding.txt

```

### Distributed CPU training and inference in a single machine

```
# Training
cd ./env_run/src
CPU_NUM=12 fleetrun --log_dir ../../fleet_logs \
                    --worker_num 4 \
                    --server_num 4 \
                    dist_cpu_train.py --config ../../user_configs/metapath2vec.yaml \
                                      --ip ../../toy_data/ip_list.txt

# Inference
CPU_NUM=12 fleetrun --log_dir ../../fleet_logs_infer \
                    --worker_num 4 \
                    --server_num 4 \
                    dist_cpu_infer.py --config ../../user_configs/metapath2vec.yaml \
                                      --ip ../../toy_data/ip_list.txt \
                                      --save_dir ./save_dist_embed/ \
                                      --infer_from ../../ckpt_custom/metapath2vec.0712/

# The node embeddings will be saved in ./save_dist_embed/

```

Note that the `worker_num` and `server_num` in inference stage should be the same as in training stage.

The training log will be saved in `../../fleet_logs`.

### Distributed CPU training with multi-machine

Suppose we perform distributed training on two machines, and deploy a server node and a trainer node on each machine. You only need to specify the ip and port list of the service node `--servers` and the ip and port list of the training node `--workers`, to perform multi-machine training.

In the following example, xx.xx.xx.xx represents machine 1, yy.yy.yy.yy represents machine 2. Then, runing the following commands in each machine.

```
# Training
cd ./env_run/src

export PADDLE_WITH_GLOO=1
export FLAGS_START_PORT=30510  # http port for multi-machine communication

CPU_NUM=12 fleetrun --log_dir ../../fleet_logs \
                    --workers "xx.xx.xx.xx:8170,yy.yy.yy.yy:8171" \
                    --servers "xx.xx.xx.xx:8270,yy.yy.yy.yy:8271" \
                    dist_cpu_train.py --config ../../user_configs/metapath2vec.yaml \
                                      --ip ../../toy_data/ip_list.txt

```
