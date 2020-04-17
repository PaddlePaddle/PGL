# PGL - Knowledge Graph Embedding

## Introduction 
This package is mainly for computing node and relation embedding of knowledge graphs efficiently.  


This package reproduce the following knowledge embedding models:
- TransE
- TransR
- RotatE

## Dataset

The dataset WN18 and FB15k are originally published by TransE paper and and be download [here](https://everest.hds.utc.fr/doku.php?id=en:transe)


## Dependencies
If you want to use the PGL-KGE in paddle, please install following packages.
- paddlepaddle>=1.7
- pgl


## Experiment results
FB15k dataset

|  Models  |Mean Rank|  Mrr  | Hits@1 | Hits@3 | Hits@10 | MR@filter| Hits10@filter| 
|----------|-------|-------|--------|--------|---------|---------|---------|
| TransE| 214 | --   | --     | --  | 0.491   | 118 | 0.668|
| TransR| 202 | --   | --     | --  | 0.502   | 115 | 0.683|
| RotatE| 156| --   | --     | --  | 0.498   | 52 | 0.710|

WN18 dataset

|  Models  |Mean Rank|  Mrr  | Hits@1 | Hits@3 | Hits@10 | MR@filter| Hits10@filter| 
|----------|-------|-------|--------|--------|---------|---------|---------|
| TransE|  257 | --   | --     | --  |  0.800  | 245 | 0.915|
| TransR|  255 | --   | --     | --  |  0.8012| 243 | 0.9371|
| RotatE|  188 | --   | --     | --  |  0.8325| 176 | 0.9601|

## References

[1]. TransE https://ieeexplore.ieee.org/abstract/document/8047276
[2]. TransR http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewFile/9571/9523
[3]. RotatE https://arxiv.org/abs/1902.10197
