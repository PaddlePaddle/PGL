# Graph based clustering using Node2vec in PGL

[Node2vec](https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf) is an algorithmic framework for representational learning on graphs. Given any graph, it can learn continuous feature representations for the nodes, which can then be used for various downstream machine learning tasks. In this work, we use it in clustering task.

## Dependencies

- paddlepaddle==2.2.0
- pgl==2.1.5

## How to run

For examples, train node2vec model on cora dataset to get the embedding vector of each node.

```sh
# train node2vec in CPU mode.
python train.py

# train node2vec in single GPU mode.
CUDA_VISIBLE_DEVICES=0 python train.py --use_cuda

```

After training finished, the embedding vector of each node will be saved in `./embedding.txt`
Then you can use the kmeans algorithm for clustering by the below commands:

```sh
python kmeans_clustering.py
```

## Hyperparameters

- conf: The model config file, default is ```./config.yaml``` . 
- epoch: Number of training epoch.
