## Masked Label Prediction: Unified Massage Passing Model for Semi-Supervised Classification

This experiment is based on stanford OGB (1.2.1) benchmark. The description of 《Masked Label Prediction: Unified Massage Passing Model for Semi-Supervised Classification》 is [avaiable here](). The steps are:

### Install environment:
``` 
    git clone https://github.com/PaddlePaddle/PGL.git
    cd PGL
    pip install -e 
    pip install -r requirements.txt
    
```
### Arxiv dataset:
  1. ```python main_arxiv.py --place 0 --log_file arxiv_baseline.txt``` to get the baseline result of arxiv dataset.
  2. ```python main_arxiv.py --place 0 --use_label_e --log_file arxiv_unipm.txt``` to get the UniPM result of arxiv dataset.
  
### Products dataset:
  1. ```python main_product.py --place 0 --log_file product_label_embedding.txt --use_label_e``` to get the UniPM result of Products dataset.
  
### Proteins dataset:
  1. ```python main_protein.py --place 0 --log_file protein_baseline.txt ``` to get the baseline result of Proteins dataset.
  2. ```python main_protein.py --place 0 --use_label_e --log_file protein_label_embedding.txt``` to get the UniPM result of Proteins dataset.
  
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
| Arxiv_UniPM        | 0.7311  ± 0.0021 | 0.7450  ± 0.0005 | 473,489 | Tesla V100 (32GB) |
| Products_baseline  | 0.8023  ± 0.0026 | 0.9286  ± 0.0017 | 1,470,905  | Tesla V100 (32GB) |
| Products_UniPM     | 0.8256  ± 0.0031 | 0.9308  ± 0.0017 | 1,475,605  | Tesla V100 (32GB) |
| Proteins_baseline  | 0.8611  ± 0.0017 | 0.9128  ± 0.0007 | 1,879,664  | Tesla V100 (32GB) |
| Proteins_UniPM     | 0.8642  ± 0.0008 | 0.9175  ± 0.0007 | 1,909,104  | Tesla V100 (32GB) |
   
