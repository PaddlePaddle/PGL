# Global Graph Pooling (Readout)

Global graph pooling is usually embedded in the end of GNNs to summarize all node features. We compare several common global pooling methods with backbone GIN:
+ Baseline: global mean pooling
+ gPool:  [Gated Graph Sequence Neural Networks](https://arxiv.org/abs/1511.05493).
+ Set2Set:[Order Matters: Sequence to sequence for Sets](https://arxiv.org/abs/1511.06391).
+ GMT: [Accurate Learning of Graph Representations with Graph Multiset Pooling](https://arxiv.org/abs/2102.11533).

For simplicity, we give the average evaluation acurracy of 10-folds cross validation.

### Datasets

The dataset can be downloaded from [here](https://github.com/weihua916/powerful-gnns/blob/master/dataset.zip).
After downloading the data，uncompress them, then a directory named `./gin_data/` can be found in current directory. 

### Dependencies

- paddlepaddle >= 2.2.0
- pgl >= 2.2.4

### How to run

For examples, use GPU to train GIN model with GMT on MUTAG dataset.
```
export CUDA_VISIBLE_DEVICES=0
python main.py --use_cuda --dataset_name MUTAG --lr 0.005 --batch_size 128 --epochs 300 --hidden_size 128 --pool_type GMT
```

### Hyperparameters

- data\_path: the root path of your dataset 
- dataset\_name: the name of the dataset. ("MUTAG", "IMDBBINARY", "IMDBMULTI", "COLLAB", "PROTEINS", "NCI1", "PTC", "REDDITBINARY", "REDDITMULTI5K")
- fold\_idx: The $fold\_idx^{th}$ fold of dataset splited. Here we use 10 fold cross-validation
- train\_eps: whether the $\epsilon$ parameter is learnable.
- pool\_type: which global pool method is selected.("GMT", "mean", "GlobalAttention", "Set2Set")

### Experiment results （Accuracy）
| |MUTAG | PROTEINS   | IMDBBINARY | IMDBMULTI |
|--|-------------|----------|------------|-----------------|
|mean | 95.14           | 68.51 | 79.61     | 57.38          |
|gPool |93.56         | 69.66 | 79.77     | 59.19          |
|Set2Set | 95.18           | 69.46 | 79.84     | 57.97          |
|GMT |93.48           | 75.65 | 80.47     | 62.41          |
