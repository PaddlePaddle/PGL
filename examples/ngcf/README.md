# Neural Graph Collaborative Filtering 

[Simplifying Graph Convolutional Networks \(NGCF\)](https://arxiv.org/pdf/1905.08108.pdf) is a powerful neural network designed for machine learning on graphs. Based on PGL, we reproduce NGCF algorithms and reach the same level of indicators as the paper.

### Datasets

The datasets contain three citation networks: Gowalla, amazon-book and Yelp2018. The details for these datasets can be found in the [paper](https://arxiv.org/abs/1905.08108).

You can download datasets from [here](https://github.com/huangtinglin/NGCF-PyTorch) and place the whole folder (e.g., "gowalla") at the root directory.

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
