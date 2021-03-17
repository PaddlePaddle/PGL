# RGCN: Modeling Relational Data with Graph Convolutional Networks

[RGCN](http://arxiv.org/abs/1703.06103) is a graph convolutional networks applied in heterogeneous graph.

Its message-passing equation is as follows:

$$
h_{i}^{(l+1)}=\sigma\left(\sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_{r}(i)} W_{r}^{(l)} h_{j}^{(l)}\right)
$$

From the equation above, we can see that there are two parts in the computation.

1, Message aggregation within each relation $r$ (edge_type).

2, Reduction that merges the results from multiple relationships.

### Datasets

Here, we use MUTAG dataset to reproduce this model. The dataset can be downloaded from [here](https://baidu-pgl.gz.bcebos.com/pgl-data/mutag_data.tar).

### Dependencies

- paddlepaddle>=2.0
- pgl>=2.1

### How to run

To train a RGCN model on MUTAG dataset, you can just run

```
export CUDA_VISIBLE_DEVICES=0

python train.py --data_path /your/path/to/mutag_data
```

If you want to train a RGCN model with multiple GPUs, you can just run with fleetrun API with `CUDA_VISIBLE_DEVICES`

```
CUDA_VISIBLE_DEVICES=0,1 fleetrun train.py --data_path /your/path/to/mutag_data
```

#### Hyperparameters

- data_path: The directory of your dataset.

- epochs: Number of epochs default (10)

- input_size: Input dimension.

- hidden_size: The hidden size for the RGCN model.

- num_class: The number of classes to be predicted.

- num_layers: The number of RGCN layers to be applied.

- num_bases: Number of basis decomposition

- seed: Random seed.

- lr: Learning rate.


### Performance

We train the RGCN model for 10 epochs and report the besst accuracy on the test dataset.

| Dataset | Accuracy   | Reported in paper |
| --- | --- | --- |
| MUTAG | 77.94% |  73.23% |
