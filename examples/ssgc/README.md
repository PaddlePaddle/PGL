# SSGC: Simple Spectral Graph Convolution

[Simple Spectral Graph Convolution \(SSGC\)](https://openreview.net/forum?id=CYO5T-YjWZV) is a powerful neural network designed for machine learning on graphs. Based on PGL, we reproduce SSGC algorithms and reach the same level of indicators as the paper in citation network benchmarks.

### Datasets

The datasets contain three citation networks: CORA, PUBMED, CITESEER. The details for these three datasets can be found in the [paper](https://arxiv.org/abs/1609.02907).

### Dependencies

- paddlepaddle >= 2.0 
- pgl >= 2.1

### Performance

We train our models for 200 epochs and report the accuracy on the test dataset.

| Dataset | Accuracy |
| --- | --- |
| Cora | 0.834 (paper: 0.830) | 
| Pubmed | 0.796 (paper: 0.804) |
| Citeseer | 0.734 (paper: 0.736) | 


### How to run

For examples, use gpu to train ssgc on cora dataset. `weight_decay` is the most import parameters.
```
# Run on GPU

# Cora 0.834 
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cora --weight_decay 5e-6  

# Citeseer 0.734 
CUDA_VISIBLE_DEVICES=0 python train.py --dataset citeseer --weight_decay  1e-4 

# Pubmed 0.796
CUDA_VISIBLE_DEVICES=0 python train.py --dataset pubmed --weight_decay  5e-6 

# Run on CPU 
CUDA_VISIBLE_DEVICES= python train.py --dataset cora
```

#### Hyperparameters

- dataset: The citation dataset "cora", "citeseer", "pubmed".
- use_cuda: Use gpu if assign use_cuda. 
