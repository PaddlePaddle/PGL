## Masked Label Prediction: Unified Massage Passing Model for Semi-Supervised Classification

This experiment is based on stanford OGB (1.2.1) benchmark. The description of 《Masked Label Prediction: Unified Massage Passing Model for Semi-Supervised Classification》 is [avaiable here](https://arxiv.org/pdf/2009.03509.pdf). The steps are:

### Note!
We propose **UniMP_large_appnp_vnode_smooth**, where we extend our base model's width by increasing ```head_num```, and make it deeper by incorporating [APPNP](https://www.in.tum.de/daml/ppnp/) . Moreover, we firstly propose a new **Attention based APPNP** and add **virtual nodes** to further improve our model's performance.

### Install environment:
``` 
    git clone https://github.com/PaddlePaddle/PGL.git
    cd PGL
    pip install -e 
    pip install -r requirements.txt
    
```
### Arxiv dataset:
  3. ```python main_arxiv_large.py --place 0 --use_label_e --log_file arxiv_unimp_large.txt``` to get the UniMP_large result of arxiv dataset.
  
  
### The **detailed hyperparameter** is:

```
Arxiv_dataset(Full Batch):              
--num_layers        3                                       
--hidden_size       128                                  
--num_heads         2                        
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
| Arxiv_UniMP_appnp_vnode_smooth        | 0.7397  ± 0.1543 | 75.0599  ± 0.092412 | 687,377 | Tesla V100 (32GB) |
   
