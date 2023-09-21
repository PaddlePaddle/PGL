# GPRGNN: Generalized PageRank Graph Neural Network

[Adaptive Universal Generalized PageRank Graph Neural Network \(GPRGNN\)](https://arxiv.org/abs/2006.07988) is a new Generalized PageRank (GPR) GNN architecture that adaptively learns the GPR weights so as to jointly optimize node feature and topological information extraction, regardless of the extent to which the node labels are homophilic or heterophilic. Based on PGL, we reproduce GPRGNN algorithms and reach the same level of indicators as the paper in citation network benchmarks.

### Datasets

The datasets contain three citation networks: CORA, PUBMED, CITESEER. The details for these three datasets can be found in the [paper](https://arxiv.org/abs/1609.02907).

### Dependencies

- paddlepaddle==2.0.0
- pgl==2.1

### Performance

We train our models for 200 epochs and report the accuracy on the test dataset.

| Dataset | Accuracy |
| --- | --- |
| Cora | ~84% | 
| Pubmed | ~78% |
| Citeseer | ~72% | 

### How to run

For examples, use gpu to train gat on cora dataset.
```
python train.py --dataset cora
```

#### Hyperparameters

- dataset: The citation dataset "cora", "citeseer", "pubmed".
- use_cuda: Use gpu if assign use_cuda. 
