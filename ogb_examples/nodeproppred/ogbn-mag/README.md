# Relational Graph Convolutional Neural Network
RGCN shows that GCN framework can be applied to modeling relational data in knowledge base, To learn more about the study of RGCN,  see [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/pdf/1703.06103.pdf) for more details.

### Datasets
In this repo, we use RGCN to deal with the ogbn-mag dataset. ogbn-mag dataset is a heterogeneous network composed of a subset of the Microsoft Academic Graph. In addition, we adopt GraphSAINT sampler in the training phase.

### Dependencies
- paddlepaddle>=1.7
- pgl>=1.1
- ogb>=1.2.0

### How to run
> CUDA_VISIBLE_DEVICES=0 python main.py --use_cude

### Hyperparameters
- epoch: Number of epochs default (40)
- use_cuda: Use gpu if assign use_cuda.
- sample_workers: The number of workers for multiprocessing subgraph sample.
- lr: Learning rate.
- batch_size: Batch size.
- hidden_size: The hidden size of the RGCN models.
- test_samples: sample num of each layers
- test_batch_size: batch_size in the test phase

### Proformance
We evaulate 8 times on the ogbn-mag dataset. Here is the result.
Dataset| Accuracy| std|
--|--|--|
ogbn-mag | 0.4727 | 0.0031 |
