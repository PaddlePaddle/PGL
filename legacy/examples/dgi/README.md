# DGI: Deep Graph Infomax 

[Deep Graph Infomax \(DGI\)](https://arxiv.org/abs/1809.10341) is a general approach for learning node representations within graph-structured data in an unsupervised manner. DGI relies on maximizing mutual information between patch representations and corresponding high-level summaries of graphs---both derived using established graph convolutional network architectures.

### Datasets

The datasets contain three citation networks: CORA, PUBMED, CITESEER. The details for these three datasets can be found in the [paper](https://arxiv.org/abs/1609.02907).

### Dependencies

- paddlepaddle>=1.6
- pgl

### Performance

We use DGI to pretrain embeddings for each nodes. Then we fix the embedding to train a node classifier.

| Dataset | Accuracy | 
| --- | --- |
| Cora | ~81% | 
| Pubmed | ~77.6% |
| Citeseer | ~71.3% |


### How to run

For examples, use gpu to train gcn on cora dataset.
```
python dgi.py --dataset cora --use_cuda
python train.py --dataset cora --use_cuda
```

#### Hyperparameters

- dataset: The citation dataset "cora", "citeseer", "pubmed".
- use_cuda: Use gpu if assign use_cuda. 
