# GCN: Graph Convolutional Networks

[Graph Convolutional Network \(GCN\)](https://arxiv.org/abs/1609.02907) is a powerful neural network designed for machine learning on graphs. Based on PGL, we reproduce GCN algorithms and reach the same level of indicators as the paper in citation network benchmarks.

### Simple example to build GCN

To build a gcn layer, one can use our pre-defined ```pgl.nn.GCNConv``` or just write a gcn layer with message passing interface.

```python


import paddle
import paddle.nn as nn

class CustomGCNConv(nn.Layer):
    def __init__(self, input_size, output_size):
        super(GCNConv, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = nn.Linear(input_size, output_size)
        self.norm = norm
        self.activation = activation

    def forward(self, graph, feature):
        norm = GF.degree_norm(graph)

        feature = self.linear(feature)

        output = graph.send_recv(feature, "sum")

        output = output * norm
        output = nn.functional.relu(output)
        return output




```

### Datasets

The datasets contain three citation networks: CORA, PUBMED, CITESEER. The details for these three datasets can be found in the [paper](https://arxiv.org/abs/1609.02907).

### Dependencies

- paddlepaddle==2.0.0
- pgl==2.1

### Performance

We train our models for 200 epochs and report the accuracy on the test dataset.

| Dataset | Accuracy |
| --- | --- |
| Cora | ~81% | 
| Pubmed | ~79% |
| Citeseer | ~71% | 


### How to run

For examples, use gpu to train gcn on cora dataset.
```
# Run on GPU
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cora

# Run on CPU 
CUDA_VISIBLE_DEVICES= python train.py --dataset cora
```

#### Hyperparameters

- dataset: The citation dataset "cora", "citeseer", "pubmed".
