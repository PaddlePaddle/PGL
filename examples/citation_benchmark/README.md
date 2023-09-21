# Easy Paper Reproduction for Citation Network ( Cora / Pubmed / Citeseer )


This page tries to reproduce all the **Graph Neural Network** paper for Citation Network (Cora/Pubmed/Citeseer) with the **public train/dev/test split**, which is the **Hello world**  dataset (**small** and **fast**) for graph neural networks. But it's very hard to achieve very high performance.



All datasets are runned with public split of  **semi-supervised** settings. And we report the averarge accuracy by running 10 times.



### Experiment Results

| Model                                                        | Cora         | Pubmed       | Citeseer     | Remarks                                                   |
| ------------------------------------------------------------ | ------------ | ------------ | ------------ | --------------------------------------------------------- |
| [Vanilla GCN (Kipf 2017)](https://openreview.net/pdf?id=SJU4ayYgl ) | 0.807(0.010) | 0.794(0.003) | 0.710(0.007) |                          -                         |
| [GAT (Veličković 2017)](https://arxiv.org/pdf/1710.10903.pdf) | 0.834(0.004) | 0.772(0.004) | 0.700(0.006) |                                -                         |
| [SGC(Wu 2019)](https://arxiv.org/pdf/1902.07153.pdf)         | 0.818(0.000) | 0.782(0.000) | 0.708(0.000) |                                 -                         |
| [APPNP (Johannes 2018)](https://arxiv.org/abs/1810.05997)    | 0.846(0.003) | 0.803(0.002) | 0.719(0.003) | Almost the same with  the results reported in Appendix E. |
| [GCNII (64 Layers, 1500 Epochs, Chen 2020)](https://arxiv.org/pdf/2007.02133.pdf) | 0.846(0.003) | 0.798(0.003) | 0.724(0.006) |            -                         |
| [SSGC (Zhu 2021)](https://openreview.net/forum?id=CYO5T-YjWZV) | 0.834(0.000) | 0.796(0.000) | 0.734(0.000) | Weight decay is important, 1e-4 for Citeseer/ 5e-6 for Cora / 5e-6  for Pubmed |
| [GATv2 (Shaked Brody 2021)](https://arxiv.org/abs/2105.14491) | 0.829(0.007) | 0.779(0.002) | 0.709(0.006) |- |
| [GPRGNN (Eli Chien 2021)](https://arxiv.org/abs/2006.07988) | 0.850(0.007) | 0.792(0.006) | 0.717(0.004) | hidden_size=64, 10 layers, weight_decay=0.0005 |
| [FAGCN (Deyu Bo 2021)](https://arxiv.org/abs/2101.00797) | 0.840(0.005) | 0.788(0.002) | 0.710(0.004) | hidden_size=16, 5 layers, 1000 epochs |

### How to run the experiments?

```shell
# Device choose
# use GPU
export CUDA_VISIBLE_DEVICES=0
# use CPU
export CUDA_VISIBLE_DEVICES=

# Experimental API
# If you want to try MultiGPU-FullBatch training. Run the following code instead.
# This will only speed up models that have more computation on edges.
# For example, the TransformerConv in [Yun 2020](https://arxiv.org/abs/2009.03509).

CUDA_VISIBLE_DEVICES=0,1 python multi_gpu_train.py --conf config/transformer.yaml

# GCN
python train.py --conf config/gcn.yaml  --dataset cora
python train.py --conf config/gcn.yaml  --dataset pubmed
python train.py --conf config/gcn.yaml  --dataset citeseer

# GAT
python train.py --conf config/gat.yaml --dataset cora
python train.py --conf config/gat.yaml --dataset pubmed
python train.py --conf config/gat.yaml --dataset citeseer

# SGC
python train.py --conf config/sgc.yaml --dataset cora
python train.py --conf config/sgc.yaml --dataset pubmed
python train.py --conf config/sgc.yaml --dataset citeseer

# APPNP
python train.py --conf config/appnp.yaml --dataset cora
python train.py --conf config/appnp.yaml --dataset pubmed
python train.py --conf config/appnp.yaml --dataset citeseer

# GCNII (The original code use 1500 epochs.)
python train.py --conf config/gcnii.yaml --dataset cora --epoch 1500
python train.py --conf config/gcnii.yaml --dataset pubmed --epoch 1500
python train.py --conf config/gcnii.yaml --dataset citeseer --epoch 1500

# TransformConv + Gated Residual
python train.py --conf config/transformer.yaml --dataset cora
python train.py --conf config/transformer.yaml --dataset pubmed
python train.py --conf config/transformer.yaml --dataset citeseer

# SSGC
python train.py --conf config/ssgc.yaml --dataset cora
python train.py --conf config/ssgc.yaml --dataset pubmed
python train.py --conf config/ssgc.yaml --dataset citeseer

# GATv2
python train.py --conf config/gatv2.yaml --dataset cora
python train.py --conf config/gatv2.yaml --dataset pubmed
python train.py --conf config/gatv2.yaml --dataset citeseer

# GPRGNN
python train.py --conf config/gpr.yaml --dataset cora
python train.py --conf config/gpr.yaml --dataset pubmed
python train.py --conf config/gpr.yaml --dataset citeseer

# FAGCN
python train.py --conf config/fagcn.yaml --dataset cora --epoch 1000
python train.py --conf config/fagcn.yaml --dataset pubmed --epoch 1000
python train.py --conf config/fagcn.yaml --dataset citeseer --epoch 1000
```
