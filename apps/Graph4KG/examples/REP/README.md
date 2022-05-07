# Relation-based Embedding Propagation

Here we provide the code of our paper ***Simple and Effective Relation-based Embedding Propagation for Knowledge Representation Learning***.

## Overview

REP is a simple and effective embedding propagation method for knowledge representation learning by utilizing graph context in knowledge graphs. The key idea is to incorporate relational graph structure information into pre-trained triplet-based embeddings. Experimental results show that by enriching pre-trained triplet-based embeddings with graph context, REP can improve or maintain prediction quality with less time cost.

<h2 align="center">
<img align="center"  src="./rep_method.png" alt="rep" width = "600" height = "350">
</h2>

## How to get knowledge embeddings

Before running REP, we need to get pre-trained triplet-based embeddings. 

In our paper, for FB15k-237 and WN18RR datasets, we use [KGEmbedding-OTE](https://github.com/JD-AI-Research-Silicon-Valley/KGEmbedding-OTE) to train KG embeddings, including 5 models we use in paper. For the best model config for TransE, RotatE and DistMult, you can find it [here](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/blob/master/best_config.sh). For OTE and GC-OTE, you can find it [here](https://github.com/JD-AI-Research-Silicon-Valley/KGEmbedding-OTE). 

For ogbl-wikikg2 dataset, we use the official code and model config released by [ogb](https://github.com/snap-stanford/ogb/tree/master/examples/linkproppred/wikikg2).

We also recommend you use [Graph4KG](https://github.com/PaddlePaddle/PGL/tree/main/apps/Graph4KG) to get knowledge embeddings.

## Run REP

After getting the knowledge embeddings of the entities and relations, you can run REP method. The main code of REP can be found at `rep.py`. For WikiKG90M dataset, you can see [here](https://github.com/PaddlePaddle/PGL/tree/main/examples/kddcup2021/WikiKG90M/post_smoothing).

Here we give the best REP hyperparameters for different models and datasets.
|  Dataset   |  Model  | Hyperparameters |
|  ----  | ---- | ---- |
| FB15k-237 | REP-TransE | alpha=0.97, khop=11 |
| FB15k-237 | REP-RotatE | alpha=0.99, khop=8 |
| FB15k-237 | REP-DistMult | alpha=0.99, khop=19 |
| FB15k-237 | REP-OTE | alpha=0.94, khop=2 |
| FB15k-237 | REP-GC-OTE | alpha=0.95, khop=2 |
| WN18RR | REP-TransE | alpha=0.7, khop=16 |
|        |            | neighbor_norm=True, degree_w=0.1 |
| WN18RR | REP-RotatE | alpha=0.99, khop=1 |
| WN18RR | REP-DistMult | alpha=0.8, khop=1 |
| WN18RR | REP-OTE | alpha=0.98, khop=4 |
| WN18RR | REP-GC-OTE | alpha=0.99, khop=7 |
| obgl-wikikg2 | REP-TransE | alpha=0.8, khop=15 |
| ogbl-wikikg2 | REP-RotatE | alpha=0.9, khop=20 |
| ogbl-wikikg2 | REP-DistMult | alpha=0.98, khop=1 |
| ogbl-wikikg2 | REP-OTE | alpha=0.98, khop=20 |
| WikiKG90M | REP-TransE | alpha=0.98, khop=10 | 
| WikiKG90M | REP-RotatE | alpha=0.98, khop=10 |
| WikiKG90M | REP-DistMult | alpha=0.98, khop=3 |
| WikiKG90M | REP-OTE | alpha=0.98, khop=13 |

Other model specific hyperparamters need to be the same as those used during training.

## Citation

Please cite the following paper if you use this code in your work.

```bibtex
@inproceedings{
    wang2022rep,
    title={Simple and Effective Relation-based Embedding Propagation for Knowledge Representation Learning},
    author={HuijuanWang and SimingDai and WeiyueSu and HuiZhong and ZeyangFang and ZhengjieHuang and ShikunFeng and ZeyuChen and YuSun and DianhaiYu 
},
    booktitle={IJCAI-ECAI},
    year={2022}
}
```
