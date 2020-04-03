# Distributed metapath2vec in PGL
[metapath2vec](https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf) is a algorithm framework for representation learning in heterogeneous networks which contains multiple types of nodes and links. Given a heterogeneous graph, metapath2vec algorithm first generates meta-path-based random walks and then use skipgram model to train a language model. Based on PGL, we reproduce metapath2vec algorithm in distributed mode.


## Datasets
DBLP: The dataset contains 14376 papers (P), 20 conferences (C), 14475 authors (A), and 8920 terms (T). There are 33791 nodes in this dataset.
You can dowload datasets from [here](https://github.com/librahu/HIN-Datasets-for-Recommendation-and-Network-Embedding)

We use the ```DBLP``` dataset for example. After downloading the dataset, put them, let's say, in ```./data/DBLP/``` .

## Dependencies
- paddlepaddle>=1.6
- pgl>=1.0.0

## How to run
Before training, run the below command to do data preprocessing.
```sh
python data_process.py --data_path ./data/DBLP  --output_path ./data/data_processed
```

We adopt [PaddlePaddle Fleet](https://github.com/PaddlePaddle/Fleet) as our distributed training frameworks. ```config.yaml``` is a configure file for metapath2vec hyperparameters and ```local_config``` is a configure file for parameter servers of PaddlePaddle. By default, we have 2 pservers and 2 trainers. One can use ```cloud_run.sh``` to help startup the parameter servers and model trainers. 

For examples, train metapath2vec in distributed mode on DBLP dataset.
```sh
# train metapath2vec in distributed mode.
sh cloud_run.sh

# multiclass task example
python multi_class.py --dataset ./data/data_processed/author_label.txt --ckpt_path ./checkpoints/2000 --num_nodes 33791

```


## Hyperparameters
All the hyper parameters are saved in ```config.yaml``` file. So before training, you can open the config.yaml to modify the hyper parameters as you like.

Some important hyper parameters in config.yaml:
- **edge_path**: the directory of graph data that you want to load
- **lr**: learning rate
- **neg_num**: number of negative samples.
- **num_walks**: number of walks started from each node
- **walk_len**: walk length
- **meta_path**: meta path scheme
