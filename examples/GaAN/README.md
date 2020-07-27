# GaAN: Gated Attention Networks for Learning on Large and Spatiotemporal Graphs

[GaAN](https://arxiv.org/abs/1803.07294) is a powerful neural network designed for machine learning on graph. It introduces an gated attention mechanism. Based on PGL, we reproduce the GaAN algorithm and train the model on [ogbn-proteins](https://ogb.stanford.edu/docs/nodeprop/#ogbn-proteins).

## Datasets
The ogbn-proteins dataset will be downloaded in directory ./dataset automatically.

## Dependencies
- [paddlepaddle >= 1.6](https://github.com/paddlepaddle/paddle)
- [pgl 1.1](https://github.com/PaddlePaddle/PGL)
- [ogb 1.1.1](https://github.com/snap-stanford/ogb)

## How to run
```bash
python train.py --lr 1e-2 --rc 0 --batch_size 1024 --epochs 100
```

or
```bash
source main.sh
```

### Hyperparameters
- use_gpu: whether to use gpu or not
- mini_data: use a small dataset to test code
- epochs: number of training epochs
- lr: learning rate
- rc: regularization coefficient
- log_path: the path of log
- batch_size: the number of batch size
- heads: the number of heads of attention
- hidden_size_a: the size of query and key vectors
- hidden_size_v: the size of value vectors
- hidden_size_m: the size of projection space for computing gates
- hidden_size_o: the size of output of GaAN layer 

## Performance
We train our models for 100 epochs and report the **rocauc** on the test dataset.
|dataset|mean|std|#experiments|
|-|-|-|-|
|ogbn-proteins|0.7803|0.0073|10|
