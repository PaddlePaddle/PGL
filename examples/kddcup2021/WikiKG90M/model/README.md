# PGL-KE: Knowledge Graph for large scale Embedding
This package is mainly for computing node and relation embedding of knowledge graphs efficiently.

This package reproduce the following knowledge embedding models:

- TransE
- RotatE
- OTE

## Background
Knowledge graphs are directed multi-relational graphs about facts,
usually expressed in the form of $(h, r, t)$ triplets,
where $h$ and $t$ represent head entity and tail entity respectively,
and $r$ is the relation between head entity and tail entity.
Large encyclopedic knowledge graphs, like Wikidata and Freebase,
can provide rich structured information about entities and benefit a wide range of applications,
such as recommender systems, question answering and information retrieval.

## Example for Large Scale Eembedding Sloution

WikiKG90M in KDD Cup 2021 is a large encyclopedic knowledge graph,
which could benefit various downstream applications such as question answering and recommender systems.
Participants are invited to complete the knowledge graph by predicting missing triplets.
Recent representation learning methods have achieved great success on standard datasets like FB15k-237.

Thus, we use the PGL-KE train the advance algorithms in different domains to learn the triplets, including OTE, QuatE, RotatE and TransE.
Significantly, based on PGL-KE, we achieved the winners' solution for the KDD Cpu 2021 WikiKG90M track.


## References

[1]. [TransE: Translating embeddings for modeling multi-relational data.](https://ieeexplore.ieee.org/abstract/document/8047276)

[2]. [RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space.](https://arxiv.org/abs/1902.10197)

[3]. [Orthogonal relation transforms with graphcontext modeling for knowledge graph embedding.](https://www.aclweb.org/anthology/2020.acl-main.241.pdf)
