# FAGCN: Frequency Adaptive Graph Convolution Network

[Beyond Low-frequency Information in Graph Convolutional Networks \(FAGCN\)](https://arxiv.org/abs/2101.00797) is a novel Frequency Adaptation Graph Convolutional Networks (FAGCN) with a self-gating mechanism, which can adaptively integrate different signals in the process of message passing, learning more information beyond low-frequency information in GNNs. Based on PGL, we reproduce FAGCN algorithms and reach the same level of indicators as the paper in citation network benchmarks.

### Datasets

The datasets contain three citation networks: CORA, PUBMED, CITESEER. The details for these three datasets can be found in the [paper](https://arxiv.org/abs/1609.02907).

### Dependencies

- paddlepaddle-gpu==2.4.0
- pgl==2.2.4

### Performance

We train our models for 2000 epochs and report the accuracy on the test dataset.

<!-- 
cora 0.8519 citeseer 0.7124 pubmed 0.789
-->

| Dataset | Accuracy |
| --- | --- |
| Cora | ~85% | 
| Pubmed | ~79% |
| Citeseer | ~71% | 

### How to run

For examples, use gpu to train gat on cora dataset.
```
python train.py --dataset cora --epoch 2000
```

#### Hyperparameters

- dataset: The citation dataset "cora", "citeseer", "pubmed".
- use_cuda: Use gpu if assign use_cuda. 
