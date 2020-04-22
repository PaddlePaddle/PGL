# PGL - Knowledge Graph Embedding


This package is mainly for computing node and relation embedding of knowledge graphs efficiently.

This package reproduce the following knowledge embedding models:
- TransE
- TransR
- RotatE

### Dataset

The dataset WN18 and FB15k are originally published by TransE paper and can be download [here](https://everest.hds.utc.fr/doku.php?id=en:transe).

FB15k: [https://drive.google.com/open?id=19I3LqaKjgq-3vOs0us7OgEL06TIs37W8](https://drive.google.com/open?id=19I3LqaKjgq-3vOs0us7OgEL06TIs37W8)

WN18: [https://drive.google.com/open?id=1MXy257ZsjeXQHZScHLeQeVnUTPjltlwD](https://drive.google.com/open?id=1MXy257ZsjeXQHZScHLeQeVnUTPjltlwD)

### Dependencies

If you want to use the PGL-KG in paddle, please install following packages.
- paddlepaddle>=1.7
- pgl

### Hyperparameters

- use\_cuda: use cuda to train.
- model: pgl-kg model names. Now available for `TransE`, `TransR` and `RotatE`.
- data\_dir: the data path of dataset.
- optimizer: optimizer to run the model.
- batch\_size: batch size.
- learning\_rate:learning rate.
- epoch: epochs to run.
- evaluate\_per\_iteration: evaluate after certain epochs.
- sample\_workers: sample workers nums to prepare data.
- margin: hyper-parameter for some model.

For more hyper parameters usages, please refer the `main.py`. We also provide `run.sh` script to reproduce performance results (please download dataset in `./data` and specify the data\_dir paramter).


### How to run

For examples, use GPU to train TransR model on WN18 dataset.
(please download WN18 dataset to `./data` floder)
```
python main.py --use_cuda --model TransR --data_dir ./data/WN18
```
We also provide `run.sh` script to reproduce following performance results.

### Experiment results

Here we report the experiment results on FB15k and WN18 dataset. The evaluation criteria are MR (mean rank), Mrr (mean reciprocal rank), Hit@N (The first N hit rate). The suffix `@f` means that we filter the exists relations of entities.

FB15k dataset

| Models | MR  |  Mrr  | Hits@1 | Hits@3 | Hits@10|  MR@f |Mrr@f|Hit1@f|Hit3@f|Hits10@f|
|--------|-----|-------|--------|--------|--------|-------|-----|------|------|--------|
| TransE | 215 | 0.205 |  0.093 | 0.234  |  0.446 |   74  |0.379| 0.235| 0.453|  0.647 |
| TransR | 304 | 0.193 |  0.092 | 0.211  |  0.418 |  156  |0.366| 0.232| 0.435|  0.623 |
| RotatE | 157 | 0.270 | 0.162  | 0.303  |  0.501 |   53  |0.478| 0.354| 0.547|  0.710 |


WN18 dataset

| Models | MR  |  Mrr  | Hits@1 | Hits@3 | Hits@10|  MR@f |Mrr@f|Hit1@f|Hit3@f|Hits10@f|
|--------|-----|-------|--------|--------|--------|-------|-----|------|------|--------|
| TransE | 219 | 0.338 | 0.082  | 0.523  |  0.800 |  208  |0.463| 0.135| 0.771| 0.932  |
| TransR | 321 | 0.370 | 0.096  | 0.591  |  0.810 |  309  |0.513| 0.158| 0.941| 0.941  |
| RotatE | 167 | 0.623 | 0.476  | 0.688  |  0.830 |  155  |0.915| 0.884| 0.941| 0.957  |


## References

[1]. [TransE: Translating embeddings for modeling multi-relational data.](https://ieeexplore.ieee.org/abstract/document/8047276)

[2]. [TransR: Learning entity and relation embeddings for knowledge graph completion.](http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewFile/9571/9523)

[3]. [RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space.](https://arxiv.org/abs/1902.10197)
