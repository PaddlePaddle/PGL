# Distribute GraphSAGE in PGL

[GraphSAGE](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf) is a general inductive framework that leverages node feature
information (e.g., text attributes) to efficiently generate node embeddings for previously unseen data. Instead of training individual embeddings for each node, GraphSAGE learns a function that generates embeddings by sampling and aggregating features from a node’s local neighborhood. Based on PGL, we reproduce GraphSAGE algorithm and reach the same level of indicators as the paper in Reddit Dataset. Besides, this is an example of subgraph sampling and training in PGL.

For purpose of high scalability, we use redis as distribute graph storage solution and training graphsage against redis server.

### Datasets(Quickstart)
The reddit dataset should be downloaded from [reddit_adj.npz](https://drive.google.com/open?id=174vb0Ws7Vxk_QTUtxqTgDHSQ4El4qDHt) and [reddit.npz](https://drive.google.com/open?id=19SphVl_Oe8SJ1r87Hr5a6znx3nJu1F2J). The details for Reddit Dataset can be found [here](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf).

- reddit.npz: https://drive.google.com/open?id=19SphVl_Oe8SJ1r87Hr5a6znx3nJu1F2J
- reddit_adj.npz: https://drive.google.com/open?id=174vb0Ws7Vxk_QTUtxqTgDHSQ4El4qDHt

Download `reddit.npz` and `reddit_adj.npz` into `data` directory for further preprocessing.

### Dependencies

```sh
pip install -r requirements.txt
```

### How to run

#### 1. Preprocessing and start reddit data service

```sh
pushd ./redis_setup
    /bin/bash ./before_hook.sh
popd
```

#### 2. training GraphSAGE model

```sh
sh ./cloud_run.sh
```

