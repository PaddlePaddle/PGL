# Distributed Deepwalk in PGL
[Deepwalk](https://arxiv.org/pdf/1403.6652.pdf) is an algorithmic framework for representational learning on graphs. Given any graph, it can learn continuous feature representations for the nodes, which can then be used for various downstream machine learning tasks. Based on PGL, we reproduce distributed deepwalk algorithms and reach the same level of indicators as the paper.

## Datasets
The datasets contain two networks: [BlogCatalog](http://socialcomputing.asu.edu/datasets/BlogCatalog3). 
## Dependencies
- paddlepaddle>=2.0rc
- pgl>=2.0

## How to run
We adopt [PaddlePaddle Fleet](https://github.com/PaddlePaddle/Fleet) as our distributed training frameworks ```config.yaml``` is config file for deepwalk hyperparameter. In distributed CPU mode, we have 2 pservers and 2 trainers. We can use ```fleetrun``` to help you startup the parameter servers and model trainers. 

For examples, train deepwalk mode on BlogCataLog dataset.
```sh
# train deepwalk in CPU mode.
python train.py
# train deepwalk in single GPU mode.
CUDA_VISIBLE_DEVICES=0 python train.py --use_cuda
# train deepwalk in multiple GPU mode.
CUDA_VISIBLE_DEVICES=0,1 fleetrun train_distributed_gpu.py
# train deepwalk in distributed CPU mode, log can be found in ./log.
python -m paddle.distributed.launch --server_num=2 --trainer_num=2 train_distributed_cpu.py

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
BlogCatalog|distributed deepwalk|multi-label classification|MacroF1|0.233|0.211
