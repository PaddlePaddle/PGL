# STGCN: Spatio-Temporal Graph Convolutional Network

[Spatio-Temporal Graph Convolutional Network \(STGCN\)](https://arxiv.org/pdf/1709.04875.pdf) is a novel deep learning framework to tackle time series prediction problem. Based on PGL, we reproduce STGCN algorithms to predict new confirmed patients in some cities with the historical immigration records.

### Datasets

You can make your customized dataset by the following format:

* input.csv: Historical immigration records with shape of [num\_time\_steps * num\_cities].

* output.csv: New confirmed patients records with shape of [num\_time\_steps * num\_cities].

* W.csv: Weighted Adjacency Matrix with shape of [num\_cities * num\_cities].

* city.csv: Each line is a number and the corresponding city name.

### Dependencies

- paddlepaddle 1.6
- pgl 1.0.0

### How to run

For examples, use gpu to train STGCN on your dataset.
```
python main.py --use_cuda --input_file dataset/input.csv --label_file dataset/output.csv --adj_mat_file dataset/W.csv --city_file dataset/city.csv 
```

#### Hyperparameters

- n\_route: Number of city.
- n\_his: "n\_his" time steps of previous observations of historical immigration records.
- n\_pred: Next "n\_pred" time steps of New confirmed patients records.
- Ks: Number of GCN layers.
- Kt: Kernel size of temporal convolution.
- use\_cuda: Use gpu if assign use\_cuda. 
