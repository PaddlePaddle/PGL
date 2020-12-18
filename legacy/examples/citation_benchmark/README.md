# Easy Paper Reproduction for Citation Network (Cora/Pubmed/Citeseer)



This page tries to reproduce all the **Graph Neural Network** paper for Citation Network (Cora/Pubmed/Citeseer), which is the **Hello world**  dataset (**small** and **fast**) for graph neural networks. But it's very hard to achieve very high performance.



All datasets are runned with public split of  **semi-supervised** settings. And we report the averarge accuracy by running 10 times.



# Experiment Results

| Model                                                        | Cora         | Pubmed       | Citeseer     | Remarks                                                   |
| ------------------------------------------------------------ | ------------ | ------------ | ------------ | --------------------------------------------------------- |
| [Vanilla GCN (Kipf 2017)](https://openreview.net/pdf?id=SJU4ayYgl ) | 0.807(0.010) | 0.794(0.003) | 0.710(0.007) |                                                           |
| [GAT (Veličković 2017)](https://arxiv.org/pdf/1710.10903.pdf) | 0.834(0.004) | 0.772(0.004) | 0.700(0.006) |                                                           |
| [SGC(Wu 2019)](https://arxiv.org/pdf/1902.07153.pdf)         | 0.818(0.000) | 0.782(0.000) | 0.708(0.000) |                                                           |
| [APPNP (Johannes 2018)](https://arxiv.org/abs/1810.05997)    | 0.846(0.003) | 0.803(0.002) | 0.719(0.003) | Almost the same with  the results reported in Appendix E. |
| [GCNII (64 Layers, 1500 Epochs, Chen 2020)](https://arxiv.org/pdf/2007.02133.pdf) | 0.846(0.003) | 0.798(0.003) | 0.724(0.006) |                                                           |





How to run the experiments?



```shell
# Device choose
export CUDA_VISIBLE_DEVICES=0
# GCN
python train.py --conf config/gcn.yaml --use_cuda --dataset cora
python train.py --conf config/gcn.yaml --use_cuda --dataset pubmed
python train.py --conf config/gcn.yaml --use_cuda --dataset citeseer


# GAT
python train.py --conf config/gat.yaml --use_cuda --dataset cora
python train.py --conf config/gat.yaml --use_cuda --dataset pubmed
python train.py --conf config/gat.yaml --use_cuda --dataset citeseer


# SGC (Slow version)
python train.py --conf config/sgc.yaml --use_cuda --dataset cora
python train.py --conf config/sgc.yaml --use_cuda --dataset pubmed
python train.py --conf config/sgc.yaml --use_cuda --dataset citeseer

# APPNP
python train.py --conf config/appnp.yaml --use_cuda --dataset cora
python train.py --conf config/appnp.yaml --use_cuda --dataset pubmed
python train.py --conf config/appnp.yaml --use_cuda --dataset citeseer

# GCNII (The original code use 1500 epochs.)
python train.py --conf config/gcnii.yaml --use_cuda --dataset cora --epoch 1500
python train.py --conf config/gcnii.yaml --use_cuda --dataset pubmed --epoch 1500
python train.py --conf config/gcnii.yaml --use_cuda --dataset citeseer --epoch 1500
```


