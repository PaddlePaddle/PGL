<h2 align="center">Graph4KG: A PaddlePaddle-based toolkit for Knowledge Representation Learning</h2>
<p align="center">
  <a href="https://ogb.stanford.edu/docs/lsc/wikikg90mv2/"><img src="https://img.shields.io/badge/KDD--CUP-2021-brightgreen"></a>
  <a href="https://arxiv.org/abs/2107.01892"><img src="http://img.shields.io/badge/Paper-PDF-59d.svg"></a> 
  </a>
</p>

## Overview

Graph4KG is a flexible framework to learn embeddings of entities and relations in KGs, which supports training on massive KGs. The features are as follows:

<h2 align="center">
<img align="center"  src="./architecture.png" alt="architecture" width = "600" height = "225">
</h2>

- Batch Pre-loading. This overlaps the time of loading batch data for next step and GPU computations of current step.
- Storage and Computation Separation. Entity embeddings are stored on the disk and loaded in the mmap mode, while computations are conducted with GPUs. 
- Asynchroneous Gradient Update. This also overlaps the computation time and gradient update time. In this case, there is at most four-step delay for gradient update. As KGs are always sparse, this asynchrony will not hurt performance.

Besides, it provides [the 1st place solution](https://ogb.stanford.edu/kddcup2021/results/#final_wikikg90m) in KDD Cup 2021.


## Requirements

 - paddlepaddle-gpu>=2.3rc
 - pgl
 - ogb==1.3.1 (optional for wikikg2 and WikiKG90M)

paddle建议使用最新develop版本。

## Models

- [x] TransE 
- [x] DistMult
- [x] ComplEx
- [x] RotatE
- [x] OTE

You can implement your score function in ```models/score_func.py```. Besides shallow methods, CNN and GNN based methods are coming soon.

**Negative Sampling**: Negative samples are constructed by randomly replacing head or tail entities with other entities. Here we implement three uniform sampling strategies.
- ``full``: Randomly sample entities from all entities in KGs.
- ``batch``: Randomly sample entities from entities arises in the same batch.
- ``chunk``: Randomly sample entities from all entities in KGs. Besides, triplets in a batch are divided into K chunks and each chunk shares the same collection of negative samples.

**Dimension**: ``embed_dim`` in ``config.py`` denoteds the dimension of real embeddings. Graph4KG will assign entity embeddings' dimension as ``embed_dim * 2`` for complex methods like RotatE and ComplEx, and as ``embed_dim * 4`` for quaternion methods like QuatE.

## Datasets

- [x] FB15k
- [x] FB15k-237
- [x] WN18
- [x] WN18RR
- [x] ogbl-wikikg2
- [x] WikiKG90M

Furthermore, other datasets formated as follows per line are also supported. You can add such new dataset in ```dataset/reader.py```.
```text
HEAD_ENTITY\tRELATION\tTAIL_ENTITY\n
```

## Examples

Scripts of different training settings are provided, including 
- [x] single-GPU
- [x] mix-CPU-GPU + async-update

```bash
# download datasets
sh examples/download.sh

# FB15k
sh examples/fb15k.sh

# FB15k-237
sh examples/fb15k237.sh

# WN18
sh examples/wn18.sh

# WN18RR
sh examples/wn18rr.sh

# WikiKG90M
sh examples/wikikg90m.sh
```

## Results

### MRR of single GPU version

| Model | FB15k | FB15k-237 | WN18 | WN18RR |
| --- | --- | --- | --- | --- |
| TransE | 0.655 | 0.316 | 0.571 | 0.189 |
| DistMult | 0.746 | 0.322 | 0.823 | 0.441 | 
| ComplEx | 0.808 | 0.324 | 0.922 | 0.464 | 
| RotatE | 0.736 | 0.225 | 0.947 | 0.469 | 
| OTE | 0.617 | 0.299 | 0.812 | 0.466 | 

### MRR of Mix CPU and GPU version

| Model | FB15k | FB15k-237 | WN18 | WN18RR |
| --- | --- | --- | --- | --- |
| TransE | 0.648 | 0.315 | 0.568| 0.187 |
| DistMult | 0.744 | 0.305 | 0.822 | 0.441 | 
| ComplEx | 0.789 | 0.312 | 0.925 | 0.464 | 
| RotatE | 0.589 | 0.286 | 0.943 |  0.463 |
| OTE | 0.512 | 0.297 | 0.656 | 0.302 | 

### MRR of WikiKG90M
| Model | MRR |
| --- | --- |
| TransE | 0.85 |
| RotatE | 0.88 |
| OTE | 0.89 |
