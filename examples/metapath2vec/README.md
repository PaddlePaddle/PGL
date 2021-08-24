# metapath2vec: Scalable Representation Learning for Heterogeneous Networks

[metapath2vec](https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf) is a algorithm framework for representation learning in heterogeneous networks which contains multiple types of nodes and links. Given a heterogeneous graph, metapath2vec algorithm first generates meta-path-based random walks and then use skipgram model to train a language model. Based on PGL, we reproduce metapath2vec algorithm using PGL graph engine for scalable representation learning.


## Dependencies

- paddlepaddle>=2.1.0

- pgl>=2.1.4

- OpenMPI==1.4.1

## Datasets

You can download datasets from [here](https://ericdongyx.github.io/metapath2vec/m2v.html).

We use the "aminer" data for example. After downloading the aminer data, put them, let's say, in `./data/net_aminer/`. We also need to move the `label/` directory to `./data/` directory.

## Data preprocessing

After downloading the dataset, run the folowing command to preprocess the data:

```
python data_preprocess.py --config config.yaml
```

## Hyperparameters

All the hyper parameters are saved in `config.yaml` file. So before training, you can open the `config.yaml` to modify the hyper parameters as you like.

## PGL Graph Engine Launching

Now we support distributed loading graph data using **PGL Graph Engine**. We also develop a simple tutorial to show how to launch a graph engine, please refer to [here](../../tutorials/working_with_distributed_graph_engine.ipynb).

To launch a distributed graph service, please follow the steps below.

### IP address setting

The first step is to set the IP list for each graph server. Each IP address with port represents a server. In `ip_list.txt` file, we set up 4 ip addresses as follow for demo:

```
127.0.0.1:8553
127.0.0.1:8554
127.0.0.1:8555
127.0.0.1:8556
```

### Launching Graph Engine by OpenMPI

Before launching the graph engine, you should set up the below hyper-parameters in `config.yaml`:

```
etype2files: "p2a:./graph_data/paper2author_edges.txt,p2c:./graph_data/paper2conf_edges.txt"
ntype2files: "p:./graph_data/node_types.txt,a:./graph_data/node_types.txt,c:./graph_data/node_types.txt"
symmetry: True
shard_num: 100
```

Then, we can launch the graph engine with the help of OpenMPI.

```
mpirun -np 4 python -m pgl.distributed.launch --ip_config ./ip_list.txt --conf ./config.yaml --mode mpi --shard_num 100
```

### Launching Graph Engine manually

If you didn't install OpenMPI, you can launch the graph engine manually. 

Fox example, if we want to use 4 servers, we should run the following command separately on 4 terminals.

```
# terminal 3
python -m pgl.distributed.launch --ip_config ./ip_list.txt --conf ./config.yaml --shard_num 100 --server_id 3

# terminal 2
python -m pgl.distributed.launch --ip_config ./ip_list.txt --conf ./config.yaml --shard_num 100 --server_id 2

# terminal 1
python -m pgl.distributed.launch --ip_config ./ip_list.txt --conf ./config.yaml --shard_num 100 --server_id 1

# terminal 0
python -m pgl.distributed.launch --ip_config ./ip_list.txt --conf ./config.yaml --shard_num 100 --server_id 0
```

Note that the `server_id` of 0 should be the last one to be launched.


## Training

After successfully launching the graph engine, you can run the below command to train the model.

```
export CUDA_VISIBLE_DEVICES=0
python train.py --config ./config.yaml --ip ./ip_list.txt
```

Note that the trained model will be saved `./ckpt_custom/$task_name/`
