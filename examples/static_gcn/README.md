# PGL Examples for GCN with StaticGraphWrapper

[Graph Convolutional Network \(GCN\)](https://arxiv.org/abs/1609.02907) is a powerful neural network designed for machine learning on graphs. Based on PGL, we reproduce GCN algorithms and reach the same level of indicators as the paper in citation network benchmarks. 

However, different from the reproduction in **examples/gcn**, we use `pgl.graph_wrapper.StaticGraphWrapper` to preload the graph data into gpu or cpu memories which achieves better performance on speed.

### Datasets

The datasets contain three citation networks: CORA, PUBMED, CITESEER. The details for these three datasets can be found in the [paper](https://arxiv.org/abs/1609.02907).

### Dependencies

- paddlepaddle>=1.6
- pgl

### Performance

We train our models for 200 epochs and report the accuracy on the test dataset.


| Dataset | Accuracy | epoch time | examples/gcn | Improvement |
| --- | --- | --- | --- | --- |
| Cora | ~81% | 0.0047s | 0.0104s | 2.21x |
| Pubmed | ~79% | 0.0049s |0.0154s | 3.14x |
| Citeseer | ~71% | 0.0045s |0.0177s | 3.93x |


### How to run

For examples, use gpu to train gcn on cora dataset.
```sh
python train.py --dataset cora --use_cuda
```

#### Hyperparameters

- dataset: The citation dataset "cora", "citeseer", "pubmed".
- use_cuda: Use gpu if assign use_cuda. 
