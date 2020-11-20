## Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification

This experiment is based on stanford OGB (1.2.1) benchmark. The description of 《Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification》 is [avaiable here](https://arxiv.org/pdf/2009.03509.pdf). The steps are:

### Note!
We propose **UniMP_large**, where we extend our base model's width by increasing ```head_num```, and make it deeper by incorporating [APPNP](https://www.in.tum.de/daml/ppnp/) . Moreover, we firstly propose a new **Attention based APPNP** to further improve our model's performance.

To_do list:
- [x] UniMP_large in Arxiv
- [ ] UniMP_large in Products
- [ ] UniMP_large in Proteins
- [ ] UniMP_xxlarge

### Install environment:
``` 
    git clone https://github.com/PaddlePaddle/PGL.git
    cd PGL
    pip install -e 
    pip install -r requirements.txt
    
```
### Arxiv dataset:
  1. ```python main_arxiv.py --place 0 --log_file arxiv_baseline.txt``` to get the baseline result of arxiv dataset.
  2. ```python main_arxiv.py --place 0 --use_label_e --log_file arxiv_unimp.txt``` to get the UniMP result of arxiv dataset.
  3. ```python main_arxiv_large.py --place 0 --use_label_e --log_file arxiv_unimp_large.txt``` to get the UniMP_large result of arxiv dataset.
  
### Products dataset:
  1. ```python main_product.py --place 0 --log_file product_unimp.txt --use_label_e``` to get the UniMP result of Products dataset.
  
### Proteins dataset:
  1. ```python main_protein.py --place 0 --log_file protein_baseline.txt ``` to get the baseline result of Proteins dataset.
  2. ```python main_protein.py --place 0 --use_label_e --log_file protein_unimp.txt``` to get the UniMP result of Proteins dataset.
  
### The **detailed hyperparameter** is:

```
Arxiv_dataset(Full Batch):          Products_dataset(NeighborSampler):          Proteins_dataset(Random Partition):
--num_layers        3               --num_layers                3               --num_layers                7                   
--hidden_size       128             --hidden_size               128             --hidden_size               64               
--num_heads         2               --num_heads                 4               --num_heads                 4
--dropout           0.3             --dropout                   0.3             --dropout                   0.1
--lr                0.001           --lr                        0.001           --lr                        0.001
--use_label_e       True            --use_label_e               True            --use_label_e               True
--label_rate        0.625           --label_rate                0.625           --label_rate                0.5 
--weight_decay.     0.0005
```

### Reference performance for OGB:

| Model              |Test Accuracy    |Valid Accuracy   | Parameters    | Hardware |
| ------------------ |--------------   | --------------- | -------------- |----------|
| Arxiv_baseline     | 0.7225  ± 0.0015 | 0.7367  ± 0.0012 | 468,369  | Tesla V100 (32GB) |
| Arxiv_UniMP        | 0.7311  ± 0.0021 | 0.7450  ± 0.0005 | 473,489 | Tesla V100 (32GB) |
| Arxiv_UniMP_large        | 0.7379  ± 0.0014 | 0.7475  ± 0.0008 | 1,162,515 | Tesla V100 (32GB) |
| Products_baseline  | 0.8023  ± 0.0026 | 0.9286  ± 0.0017 | 1,470,905  | Tesla V100 (32GB) |
| Products_UniMP     | 0.8256  ± 0.0031 | 0.9308  ± 0.0017 | 1,475,605  | Tesla V100 (32GB) |
| Proteins_baseline  | 0.8611  ± 0.0017 | 0.9128  ± 0.0007 | 1,879,664  | Tesla V100 (32GB) |
| Proteins_UniMP     | 0.8642  ± 0.0008 | 0.9175  ± 0.0007 | 1,909,104  | Tesla V100 (32GB) |
   
