# Relation-based Embedding Propagation

Here we provide the code of our paper ***Simple and Effective Relation-based Embedding Propagation for Knowledge Representation Learning***.

## Overview

REP is a simple and effective embedding propagation method for knowledge representation learning by utilizing graph context in knowledge graphs. The key idea is to incorporate relational graph structure information into pre-trained triplet-based embeddings. Experimental results show that by enriching pre-trained triplet-based embeddings
with graph context, REP can improve or maintain prediction quality with less time cost.

<h2 align="center">
<img align="center"  src="./rep_method.png" alt="rep" width = "600" height = "350">
</h2>

## How to get knowledge embeddings

Before running REP, we need to get pre-trained triplet-based embeddings. 

In our paper, for FB15k-237 and WN18RR datasets, we use [KGEmbedding-OTE](https://github.com/JD-AI-Research-Silicon-Valley/KGEmbedding-OTE) to train KG embeddings, including 5 models we use in paper. For the best model config for TransE, RotatE and DistMult, we can find it [here](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/blob/master/best_config.sh). For OTE and GC-OTE, we can find it [here](https://github.com/JD-AI-Research-Silicon-Valley/KGEmbedding-OTE). 

For ogbn-wikikg2 dataset, we use the official code and model config released by [ogb](https://github.com/snap-stanford/ogb/tree/master/examples/linkproppred/wikikg2).

We also recommend you use [Graph4KG](https://github.com/PaddlePaddle/PGL/tree/main/apps/Graph4KG) to get knowledge embeddings.

## Run REP

After getting the knowledge embeddings of the entities and relations, we can run REP method. The main code of REP can be found at `rep.py`. For Wikikg90m dataset, you can see [here](https://github.com/PaddlePaddle/PGL/tree/main/examples/kddcup2021/WikiKG90M/post_smoothing).

Here we give the best REP hyperparameters for different models and datasets.


## Citation
