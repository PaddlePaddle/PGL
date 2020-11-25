## Masked Label Prediction: Unified Massage Passing Model for Semi-Supervised Classification

This experiment is based on stanford OGB (1.2.1) benchmark. The description of 《Masked Label Prediction: Unified Massage Passing Model for Semi-Supervised Classification》 is [avaiable here](https://arxiv.org/pdf/2009.03509.pdf). In this directory, we provide more improvement tricks.

### More Tricks for Improvemnts for OGBN-Proteins
- **Virtual Nodes**
    - Virtual Nodes is used in [Graph Classification task](https://arxiv.org/pdf/1708.04357.pdf). We firstly adopt this module in node Classification task and facilitate fast convergence and achieve higher score.
- **Attention & Smooth Aggregate**
    - The attention mechanism can capture important information, but the expression is too extreme. 
    - We use an additional average aggregate and then add the two results together proportionally, to smooth the output of the aggregater.
- **Attention based APPNP**
    - We firstly propose a new attention based APPNP. Instead of the graph structure in [APPNP](https://www.in.tum.de/daml/ppnp/), the attention mastrix in used for message passing in the last layer.
- **Gated based APPNP**
    - The original APPNP use a fix teleport probability alpha. Get inspired from LSTM gate, we use a gate to learn the teleport probability in every appnp layer.


### Install environment:
``` 
    git clone https://github.com/PaddlePaddle/PGL.git
    cd PGL
    pip install -e 
    pip install -r requirements.txt
    
```
### Arxiv dataset:
  ```python main_arxiv_appnp_vnode_smooth.py --place 0 --use_label_e --log_file arxiv_unimp.txt``` to get the UniMP result with above tricks of arxiv dataset.
  
  
### The **detailed hyperparameter** is:

```
Arxiv_dataset(Full Batch):              
--num_layers        3                                       
--hidden_size       100                                  
--num_heads         3                        
--dropout           0.3                      
--attn_dropout      0.1
--lr                0.001                    
--use_label_e       True                     
--label_rate        0.625                    
--weight_decay.     0.0005
--epochs            1500
--num_vnode         3
```

### Reference performance for OGB:

| Model              |Test Accuracy    |Valid Accuracy   | Parameters    | Hardware |
| ------------------ |--------------   | --------------- | -------------- |----------|
| Arxiv_baseline     | 0.7225  ± 0.0015 | 0.7367  ± 0.0012 | 468,369  | Tesla V100 (32GB) |
| Arxiv_UniMP        | 0.7311  ± 0.0021 | 0.7450  ± 0.0005 | 473,489 | Tesla V100 (32GB) |
| Arxiv_UniMP_large        | 0.7379  ± 0.0014 | 0.7475  ± 0.0008 | 1,162,515 | Tesla V100 (32GB) |
| Arxiv_UniMP_appnp_vnode_smooth        | 0.7397  ± 0.0015 | 0.7505  ± 0.0009 | 687,377 | Tesla V100 (32GB) |
   
