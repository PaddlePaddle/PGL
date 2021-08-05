# LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation 

[LightGCN](https://arxiv.org/pdf/2002.02126.pdf) is a simple but effective neural network designed for machine learning on graphs. Based on PGL, we reproduce LightGCN algorithms and reach the same level of indicators as the paper.

### Datasets

The datasets contain three citation networks: Gowalla, amazon-book. The details for these datasets can be found in the [paper](https://arxiv.org/abs/2002.02126).

You can download datasets from [here](https://github.com/kuandeng/LightGCN) and place the whole folder (e.g., "gowalla") at the root directory.

### Dependencies

- paddlepaddle >= 2.0 
- pgl >= 2.1

### How to run

For examples, use gpu to train lightgcn on gowalla dataset.
```
# Run on GPU
python train.py --dataset gowalla
```

### Contributor

AI Studio用户: 白马非马, Maple天狗
