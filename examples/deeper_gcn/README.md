# DeeperGCN: All You Need to Train Deeper GCNs

see more information in https://arxiv.org/pdf/2006.07739.pdf


### Datasets

The datasets contain three citation networks: CORA, PUBMED, CITESEER. The details for these three datasets can be found in the [paper](https://arxiv.org/abs/1609.02907).

### Dependencies

- paddlepaddle>=1.6
- pgl

### Performance

We train our models for 200 epochs and report the accuracy on the test dataset.

| Dataset | Accuracy |
| --- | --- |
| Cora | ~77% | 

### How to run

For examples, use gpu to train gat on cora dataset.
```
python train.py --dataset cora --use_cuda
```

#### Hyperparameters

- dataset: The citation dataset "cora", "citeseer", "pubmed".
- use_cuda: Use gpu if assign use_cuda. 
