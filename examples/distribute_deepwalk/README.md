# Distributed Deepwalk in PGL
[Deepwalk](https://arxiv.org/pdf/1403.6652.pdf) is an algorithmic framework for representational learning on graphs. Given any graph, it can learn continuous feature representations for the nodes, which can then be used for various downstream machine learning tasks. Based on PGL, we reproduce distributed deepwalk algorithms and reach the same level of indicators as the paper.

## Datasets
The datasets contain two networks: [BlogCatalog](http://socialcomputing.asu.edu/datasets/BlogCatalog3). 
## Dependencies
- paddlepaddle>=1.6
- pgl>=1.0

## How to run

We adopt [PaddlePaddle Fleet](https://github.com/PaddlePaddle/Fleet) as our distributed training frameworks ```pgl_deepwalk.cfg``` is config file for deepwalk hyperparameter and ```local_config``` is a config file for parameter servers. By default, we have 2 pservers and 2 trainers. We can use ```cloud_run.sh``` to help you startup the parameter servers and model trainers. 

For examples, train deepwalk in distributed mode on BlogCataLog dataset.
```sh
# train deepwalk in distributed mode.
sh cloud_run.sh

# multiclass task example
python3 multi_class.py --use_cuda --ckpt_path ./model_path/4029 --epoch 1000

```

## Hyperparameters
- dataset: The citation dataset "BlogCatalog".
- hidden_size: Hidden size of the embedding. 
- lr: Learning rate. 
- neg_num: Number of negative samples.
- epoch: Number of training epoch.

### Experiment results
Dataset|model|Task|Metric|PGL Result|Reported Result 
--|--|--|--|--|--
BlogCatalog|distributed deepwalk|multi-label classification|MacroF1|0.233|0.211
