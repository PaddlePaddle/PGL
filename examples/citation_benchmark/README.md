# Easy Paper Reproduction for Citation Network (Cora/Pubmed/Citeseer)



This page tries to reproduce all the **Graph Neural Network** paper for Citation Network (Cora/Pubmed/Citeseer), which is the **Hello world**  dataset (**small** and **fast**) for graph neural networks. But it's very hard to achieve very high performance.



All datasets are runned with public split of  **semi-supervised** settings.



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
python train.py --conf config/appnp.yaml --use_cuda --dataset cora --epoch 1500
python train.py --conf config/appnp.yaml --use_cuda --dataset pubmed --epoch 1500
python train.py --conf config/appnp.yaml --use_cuda --dataset citeseer --epoch 1500
```



