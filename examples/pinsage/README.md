# PinSage

[PinSage](https://arxiv.org/abs/1806.01973) combines efficient random walks and graph convolutions to generate embeddings of nodes (i.e., items) that incorporate both graph structure as well as node feature information.


### Datasets
The reddit dataset should be downloaded from the following links and placed in the directory ```pgl.data```. The details for Reddit Dataset can be found [here](https://cs.stanford.edu/people/jure/pubs/pinsage-nips17.pdf).

- reddit.npz https://drive.google.com/open?id=19SphVl_Oe8SJ1r87Hr5a6znx3nJu1F2J
- reddit_adj.npz: https://drive.google.com/open?id=174vb0Ws7Vxk_QTUtxqTgDHSQ4El4qDHt


### Dependencies

- paddlepaddle>=2.0
- pgl

### How to run

To train a PinSage model on Reddit Dataset, you can just run

```
 python train.py  --epoch 10  --normalize --symmetry     
```

If you want to train a PinSage model with multiple GPUs, you can just run with fleetrun API with `CUDA_VISIBLE_DEVICES`

```
CUDA_VISIBLE_DEVICES=0,1 fleetrun train.py --epoch 10 --normalize --symmetry 
```


#### Hyperparameters

- epoch: Number of epochs default (10)
- normalize: Normalize the input feature if assign normalize.
- sample_workers: The number of workers for multiprocessing subgraph sample.
- lr: Learning rate.
- symmetry: Make the edges symmetric if assign symmetry.
- batch_size: Batch size.
- samples: The max neighbors for each layers hop neighbor sampling. (default: [30, 20])
- top_k: the top k nodes should be reseved.
- hidden_size: The hidden size of the PinSage models.


### Performance

We train our models for 10 epochs and report the accuracy on the test dataset.


| Aggregator | Accuracy |
| --- |   ---  |
| SUM | 91.36% |
