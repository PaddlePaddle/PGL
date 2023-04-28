# ChebNetII: Convolutional Neural Networks on Graphs with Chebyshev Approximation, Revisited

[Convolutional Neural Networks on Graphs with Chebyshev Approximation, Revisited \(ChebNetII\)](https://arxiv.org/abs/2202.03580) is a new GNN model based on Chebyshev interpolation, which enhances the original Chebyshev polynomial approximation while reducing the Runge phenomenon. This repository contains a Paddle Graph Learning (PGL) implementation of ChebNetII algorithms and reach the same level of indicators as the paper.

### Dataset

The datasets contain three small citation networks: Cora, CiteSeer, Pubmed from the [paper](https://arxiv.org/abs/1609.02907) and two large citation networks: ogbn-arxiv, ogbn-papers100m from [OGB](https://ogb.stanford.edu/).

### Dependencies

- paddlepaddle==2.3.2
- pgl==2.2.4
- torch==1.11.0 (only for OGB dataset)
- ogb==1.3.5 (only for OGB dataset)


### How to run

For examples, use gpu to train ChebNetII on small citation dataset.

```sh
python train.py  --dataset cora --lr 0.01  --weight_decay 0.0005 --dprate 0.0
python train.py  --dataset citeseer  --lr 0.01  --weight_decay 0.0005 --prop_wd 0.0
python train.py  --dataset pubmed   --lr 0.01  --weight_decay 0.0 --prop_wd 0.0 --dprate 0.0
```

For OGB dataset, you need to run the preprocessing code and then run the this script.

```sh
python large_train.py --data arxiv --lr 0.01 --dropout 0.5 --hidden 512 --pro_lr 0.01 --pro_wd 0.0
```

### Performance

We train ChebNetII for 10 runs and report the accuracy on the test dataset.

| Dataset | Accuracy |
| --- | --- |
| Cora | 88.87(1.19) |
| CiteSeer | 80.24(1.22)|
| PubMed | 89.05(0.53) |
| ogbn-arxiv | 72.23(0.37) |


### Citation

```sh
@inproceedings{he2022chebnetii,
  title={Convolutional Neural Networks on Graphs with Chebyshev Approximation, Revisited},
  author={He, Mingguo and Wei, Zhewei and Wen, Ji-Rong},
  booktitle={NeurIPS},
  year={2022}
}
```
