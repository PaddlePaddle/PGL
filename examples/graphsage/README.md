# GraphSAGE: Inductive Representation Learning on Large Graphs

[GraphSAGE](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf) is a general inductive framework that leverages node feature
information (e.g., text attributes) to efficiently generate node embeddings for previously unseen data. Instead of training individual embeddings for each node, GraphSAGE learns a function that generates embeddings by sampling and aggregating features from a node’s local neighborhood. Based on PGL, we reproduce GraphSAGE algorithm and reach the same level of indicators as the paper in Reddit Dataset. Besides, this is an example of subgraph sampling and training in PGL.

### Datasets
The reddit dataset should be downloaded from the following links and placed in the directory ```pgl.data```. The details for Reddit Dataset can be found [here](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf).

- reddit.npz https://drive.google.com/open?id=19SphVl_Oe8SJ1r87Hr5a6znx3nJu1F2J
- reddit_adj.npz: https://drive.google.com/open?id=174vb0Ws7Vxk_QTUtxqTgDHSQ4El4qDHt


### Dependencies

- paddlepaddle>=2.0
- pgl

### How to run

To train a GraphSAGE model on Reddit Dataset, you can just run

```
 python train.py  --epoch 10  --normalize --symmetry     
```

If you want to train a GraphSAGE model with multiple GPUs, you can just run with fleetrun API with `CUDA_VISIBLE_DEVICES`

```
CUDA_VISIBLE_DEVICES=0,1 fleetrun train.py --epoch 10 --normalize --symmetry 
```

If you want to train a GraphSAGE model with CPU Parameters, you can just run with fleetrun API with `train_distributed_cpu.py`

```
fleetrun --worker_num 2 --server_num 2 train_distributed_cpu.py --epoch 10 --normalize --symmetry
```


#### Hyperparameters

- epoch: Number of epochs default (10)
- normalize: Normalize the input feature if assign normalize.
- sample_workers: The number of workers for multiprocessing subgraph sample.
- lr: Learning rate.
- symmetry: Make the edges symmetric if assign symmetry.
- batch_size: Batch size.
- samples: The max neighbors for each layers hop neighbor sampling. (default: [25, 10])
- hidden_size: The hidden size of the GraphSAGE models.


### Performance

We train our models for 200 epochs and report the accuracy on the test dataset.


| Aggregator | Accuracy   | Reported in paper |
| --- | --- | --- |
| Mean | 95.70% |  95.0% |
