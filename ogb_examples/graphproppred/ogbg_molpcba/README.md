## Graph Property Prediction for Open Graph Benchmark (OGB) Molpcba Dataset

[The Open Graph Benchmark (OGB)](https://ogb.stanford.edu/) is a collection of benchmark datasets, data loaders, and evaluators for graph machine learning. Here we complete the Graph Property Prediction task on Molpcba dataset using [GINE+](https://arxiv.org/pdf/2011.15069.pdf) with [APPNP](https://arxiv.org/abs/1810.05997.pdf) algorithm based on PGL.


### Requirements

paddlpaddle == 2.0.0

pgl == 2.1.2



### Results on ogbg-molpcba
Here, we demonstrate the following performance on the ogbg-molpcba dataset from Stanford Open Graph Benchmark (1.2.5)

| Model              |Test AP    |Validation AP  | Parameters    | Hardware |
| ------------------ |--------------   | --------------- | -------------- |----------|
| GINE+ w/ virtual nodes w/ APPNP     | 0.2979 ± 0.0030 | 0.3126 ± 0.0023 | 6,147,029  | Tesla V100 (32GB) |


### Reproducing results
To simply reproduce the results demonstrated above, run the following commands: 

```
export CUDA_VISIBLE_DEVICES=0

python main.py --config pcba_config.yaml
```
the results will be saved in `./logs/log.txt` file.


### Detailed hyperparameters
All the hyperparameters can be found in the `pcba_config.yaml` file. 

```
K: 3
hidden_size: 400
out_dim: 128
dropout_prob: 0.5
virt_node: True
conv_type: "gin+"
num_layers: 5
appnp_hop: 5
alpha: 0.8 

epochs: 100
batch_size: 100
lr: 0.005
num_workers: 4
shuffle: True
```

   
