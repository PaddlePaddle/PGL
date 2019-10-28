# SGC: Simplifying Graph Convolutional Networks 

[Simplifying Graph Convolutional Networks \(SGC\)](https://arxiv.org/pdf/1902.07153.pdf) is a powerful neural network designed for machine learning on graphs. Based on PGL, we reproduce SGC algorithms and reach the same level of indicators as the paper in citation network benchmarks.

### Datasets

The datasets contain three citation networks: CORA, PUBMED, CITESEER. The details for these three datasets can be found in the [paper](https://arxiv.org/abs/1609.02907).

### Dependencies

- paddlepaddle 1.5
- pgl

### Performance

We train our models for 200 epochs and report the accuracy on the test dataset.

| Dataset | Accuracy | Speed with paddle 1.5 <br> (epoch time)|
| --- | --- | ---|
| Cora | 0.818 (paper: 0.810) | 0.0015s | 
| Pubmed | 0.788 (paper: 0.789) | 0.0015s |
| Citeseer | 0.719 (paper: 0.719) | 0.0015s | 


### How to run

For examples, use gpu to train SGC on cora dataset.
```
python sgc.py --dataset cora --use_cuda
```

#### Hyperparameters

- dataset: The citation dataset "cora", "citeseer", "pubmed".
- use_cuda: Use gpu if assign use_cuda. 
