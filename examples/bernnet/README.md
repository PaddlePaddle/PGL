# BernNet: Learning Arbitrary Graph Spectral Filters via Bernstein Approximation

[Learning Arbitrary Graph Spectral Filters via Bernstein Approximation \(BernNet\)](https://arxiv.org/abs/2106.10994) is a novel graph neural network with theoretical support that provides a simple but effective scheme for designing and learning arbitrary graph spectral filters. This repository contains a Paddle Graph Learning (PGL) implementation of BernNet algorithms and reach the same level of indicators as the paper.

### Dataset

The datasets contain three small citation networks: Cora, CiteSeer, Pubmed from the [paper](https://arxiv.org/abs/1609.02907).

### Dependencies

- paddlepaddle==2.3.2
- pgl==2.2.4

### How to run

For examples, use gpu to train BernNet on citation dataset.

```sh
python train.py  --dataset cora --prop_lr 0.01 --dprate 0.0
python train.py  --dataset citeseer --prop_lr 0.01 --dprate 0.5
python train.py  --dataset pubmed --prop_lr 0.01 --dprate 0.0 --weight_decay 0.0
```

### Performance

We train ChebNetII for 10 runs and report the accuracy on the test dataset.

| Dataset | Accuracy |
| --- | --- |
| Cora | 88.97(1.56) |
| CiteSeer | 79.96(1.31)|
| PubMed | 88.93(0.43) |


### Citation

```sh
@inproceedings{he2021bernnet,
  title={BernNet: Learning Arbitrary Graph Spectral Filters via Bernstein Approximation},
  author={He, Mingguo and Wei, Zhewei and Huang, Zengfeng and Xu, Hongteng},
  booktitle={NeurIPS},
  year={2021}
}
```
