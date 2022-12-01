# Self-Attention Graph Pooling (SAGPool)

[Self-Attention Graph Pooling \(SAGPool\)](https://arxiv.org/abs/1904.08082)  is a hierarchical graph pooling methods which embed behind GNN layers to remove the nodes with low attention score. 

Reported results is the average evaluation acurracy of 10-folds cross validation.

### Datasets

The dataset can be downloaded from [here](https://github.com/weihua916/powerful-gnns/blob/master/dataset.zip).
After downloading the data，uncompress them, then a directory named `./gin_data/` can be found in current directory. 

### Dependencies

- paddlepaddle >= 2.2.0
- pgl >= 2.2.4

### How to run

For examples, use GPU to train GIN model with SAGPool on PROTEINS dataset.
```
export CUDA_VISIBLE_DEVICES=0
python main.py --use_cuda --dataset_name PROTEINS --lr 0.005 --batch_size 128 --epochs 300 --hidden_size 128 --min_score 0.001
```

### Hyperparameters

- data\_path: the root path of your dataset 
- dataset\_name: the name of the dataset. ("MUTAG", "IMDBBINARY", "IMDBMULTI", "COLLAB", "PROTEINS", "NCI1", "PTC", "REDDITBINARY", "REDDITMULTI5K")
- fold\_idx: The $fold\_idx^{th}$ fold of dataset splited. Here we use 10 fold cross-validation
- min\_score: parameter for SAGPool which indicates minimal node score. (When min\_score is not None, pool\_ratio is ignored)
- pool\_ratio: parameter for SAGPool which decides how many nodes will be removed.

### Experiment results （Accuracy）
|   | PROTEINS   | IMDBBINARY | IMDBMULTI | NCI1|
|-------------|----------|------------|-----------------|-----------------|
|mean    | 68.51  | 79.61     | 57.38      |68.32| 
|SAGPool | 76.90  | 79.77     |59.36       |73.26|
