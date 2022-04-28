# Relation-based Embedding Propagation

Here we provide the code of our paper ***Simple and Effective Relation-based Embedding Propagation for Knowledge Representation Learning***.

## Overview

REP is a simple and effective embedding propagation method for knowledge representation learning by utilizing graph context in KGs. The key idea is to incorporate relational graph structure information into pre-trained triplet-based embeddings. Experimental results show that by enriching pre-trained triplet-based embeddings
with graph context, REP can improve or maintain prediction quality with less time cost.

<h2 align="center">
<img align="center"  src="./rep_method.png" alt="rep" width = "600" height = "225">
</h2>

## Get Knowledge Embedding

Before running REP, we need to get pre-trained triplet-based embeddings. 

For FB15k-237 and WN18RR datasets, we use https://github.com/JD-AI-Research-Silicon-Valley/KGEmbedding-OTE to train KG embeddings. As for the best model config for TransE, RotatE and DistMult, we can find [here](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/blob/master/best_config.sh).

For ogbn-wikikg2 dataset, we use 


## Run REP


## Citation
