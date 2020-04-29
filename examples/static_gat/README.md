# PGL Examples for GAT with StaticGraphWrapper

[Graph Attention Networks \(GAT\)](https://arxiv.org/abs/1710.10903) is a novel architectures that operate on graph-structured data, which leverages masked self-attentional layers to address the shortcomings of prior methods based on graph convolutions or their approximations. Based on PGL, we reproduce GAT algorithms and reach the same level of indicators as the paper in citation network benchmarks.

However, different from the reproduction in **examples/gat**, we use `pgl.graph_wrapper.StaticGraphWrapper` to preload the graph data into gpu or cpu memories which achieves better performance on speed.


### Datasets

The datasets contain three citation networks: CORA, PUBMED, CITESEER. The details for these three datasets can be found in the [paper](https://arxiv.org/abs/1609.02907).

### Dependencies

- paddlepaddle>=1.6
- pgl

### Performance

We train our models for 200 epochs and report the accuracy on the test dataset.


| Dataset | Accuracy | epoch time | examples/gat | Improvement |
| --- | --- | --- | --- | --- |
| Cora | ~83% | 0.0119s | 0.0175s | 1.47x |
| Pubmed | ~78% | 0.0193s |0.0295s | 1.53x |
| Citeseer | ~70% | 0.0124s |0.0253s | 2.04x |

### How to run

For examples, use gpu to train gat on cora dataset.
```sh
python train.py --dataset cora --use_cuda
```

#### Hyperparameters

- dataset: The citation dataset "cora", "citeseer", "pubmed".
- use_cuda: Use gpu if assign use_cuda. 
