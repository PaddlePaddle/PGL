# GES:  Graph Embedding with Side Information
[Graph Embedding with Side Information](https://arxiv.org/pdf/1803.02349.pdf) is an algorithmic framework for representational learning on graphs. Given any graph, it can learn continuous feature representations for the nodes, which can then be used for various downstream machine learning tasks. Based on PGL, we reproduce ges algorithms.
## Datasets
The datasets contain two networks: [BlogCatalog](http://socialcomputing.asu.edu/datasets/BlogCatalog3). 
## Dependencies
- paddlepaddle>=1.6
- pgl>=1.0.0

## How to run

For examples, train ges on cora dataset.
```sh
# train deepwalk in distributed mode.
sh gpu_run.sh
```

## Hyperparameters
- dataset: The citation dataset "BlogCatalog".
- hidden_size: Hidden size of the embedding. 
- lr: Learning rate. 
- neg_num: Number of negative samples.
- epoch: Number of training epoch.
