# Graph Isomorphism Network (GIN)

[Graph Isomorphism Network \(GIN\)](https://arxiv.org/pdf/1810.00826.pdf) is a simple graph neural network that expects to achieve the ability as the Weisfeiler-Lehman graph isomorphism test. Based on PGL, we reproduce the GIN model.

### Datasets

The dataset can be downloaded from [here](https://github.com/weihua916/powerful-gnns/blob/master/dataset.zip).
After downloading the data，uncompress them, then a directory named `./dataset/` can be found in current directory. Note that the current directory is the root directory of GIN model.

### Dependencies

- paddlepaddle >= 2.0.0
- pgl >= 2.0

### How to run

For examples, use GPU to train GIN model on MUTAG dataset.
```
export CUDA_VISIBLE_DEVICES=0
python main.py --use_cuda --dataset_name MUTAG  --data_path ./dataset
```

### Hyperparameters

- data\_path: the root path of your dataset 
- dataset\_name: the name of the dataset
- fold\_idx: The $fold\_idx^{th}$ fold of dataset splited. Here we use 10 fold cross-validation
- train\_eps: whether the $\epsilon$ parameter is learnable.

### Experiment results （Accuracy）
| |MUTAG | COLLAB   | IMDBBINARY | IMDBMULTI |
|--|-------------|----------|------------|-----------------|
|PGL result | 90.8           | 78.6 | 76.8     | 50.8          |
|paper reuslt |90.0           | 80.0 | 75.1     | 52.3          |
