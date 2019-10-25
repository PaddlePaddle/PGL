# Distribute GraphSAGE in PGL

[GraphSAGE](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf) is a general inductive framework that leverages node feature
information (e.g., text attributes) to efficiently generate node embeddings for previously unseen data. Instead of training individual embeddings for each node, GraphSAGE learns a function that generates embeddings by sampling and aggregating features from a node’s local neighborhood. Based on PGL, we reproduce GraphSAGE algorithm and reach the same level of indicators as the paper in Reddit Dataset. Besides, this is an example of subgraph sampling and training in PGL.

For purpose of high scalability, we use redis as distribute graph storage solution and training graphsage against redis server.

### Datasets(Quickstart)
The reddit dataset should be downloaded from [reddit_adj.npz](https://drive.google.com/open?id=174vb0Ws7Vxk_QTUtxqTgDHSQ4El4qDHt) and [reddit.npz](https://drive.google.com/open?id=19SphVl_Oe8SJ1r87Hr5a6znx3nJu1F2Jthe). The details for Reddit Dataset can be found [here](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf).

Alternatively, reddit dataset has been preprocessed and packed into docker image, which can be instantly pulled using following commands.

```sh
docker pull githubutilities/reddit_redis_demo:v0.1
```

### Dependencies

```txt
- paddlepaddle>=1.6
- pgl
- scipy
- redis==2.10.6
- redis-py-cluster==1.3.6
```

### How to run

#### 1. Start reddit data service

```sh
docker run \
    --net=host \
    -d --rm \
    --name reddit_demo \
    -it githubutilities/reddit_redis_demo:v0.1 \
    /bin/bash -c "/bin/bash ./before_hook.sh && /bin/bash"
docker logs -f `docker ps -aqf "name=reddit_demo"`
```

#### 2. training GraphSAGE model

```sh
python train.py --use_cuda --epoch 10 --graphsage_type graphsage_mean --sample_workers 10
```

#### Hyperparameters

- epoch: Number of epochs default (10)
- use_cuda: Use gpu if assign use_cuda. 
- graphsage_type: We support 4 aggregator types including "graphsage_mean", "graphsage_maxpool", "graphsage_meanpool" and "graphsage_lstm".
- sample_workers: The number of workers for multiprocessing subgraph sample.
- lr: Learning rate.
- batch_size: Batch size.
- samples_1: The max neighbors for the first hop neighbor sampling. (default: 25)
- samples_2: The max neighbors for the second hop neighbor sampling. (default: 10)
- hidden_size: The hidden size of the GraphSAGE models.

### View the Code

See the code [here](distribute_graphsage_examples_code.html)
