# Distributed Node2vec in PGL
[Node2vec](https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf) is an algorithmic framework for representational learning on graphs. Given any graph, it can learn continuous feature representations for the nodes, which can then be used for various downstream machine learning tasks. Based on PGL, we reproduce distributed node2vec algorithms and reach the same level of indicators as the paper.

## Datasets
The datasets contain two networks: [BlogCatalog](http://socialcomputing.asu.edu/datasets/BlogCatalog3). 
## Dependencies
- paddlepaddle>=2.0
- pgl>=2.0

## How to run
We adopt [PaddlePaddle Fleet](https://github.com/PaddlePaddle/Fleet) as our distributed training frameworks ```config.yaml``` is a config file for node2vec hyperparameter. In distributed CPU mode, we have 2 pservers and 2 trainers. We can use ```fleetrun``` to help you startup the parameter servers and model trainers. 

For examples, train node2vec mode on BlogCataLog dataset.
```sh
# train node2vec in CPU mode.
python train.py
# train node2vec in single GPU mode.
CUDA_VISIBLE_DEVICES=0 python train.py --use_cuda
# train node2vec in multiple GPU mode.
CUDA_VISIBLE_DEVICES=0,1 fleetrun train.py --use_cuda
# train node2vec in distributed CPU mode.
CPU_NUM=10 fleetrun --worker_num 2 --server_num 2 train_distributed_cpu.py
# The output log is redirected to 'log/workerlog.0'

# multiclass task example
python multi_class.py

```

## Hyperparameters
- dataset: The citation dataset "BlogCatalog".
- conf: The model config file, default is ```./config.yaml``` . 
- epoch: Number of training epoch.

### Experiment results
Dataset|model|Task|Metric|PGL Result|Reported Result 
--|--|--|--|--|--
BlogCatalog|distributed node2vec|multi-label classification|MacroF1|0.260|0.258
