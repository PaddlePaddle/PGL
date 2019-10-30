# metapath2vec: Scalable Representation Learning for Heterogeneous Networks
[metapath2vec](https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf) is a algorithm framework for representation learning in heterogeneous networks which contains multiple types of nodes and links. Given a heterogeneous graph, metapath2vec algorithm first generates meta-path-based random walks and then use skipgram model to train a language model. Based on PGL, we reproduce metapath2vec algorithm.


## Datasets
You can dowload datasets from [here](https://ericdongyx.github.io/metapath2vec/m2v.html)

We use the "aminer" data for example. After downloading the aminer data, put them, let's say, in ./data/net_aminer/ . We also need to put "label/" directory in ./data/.

## Dependencies
- paddlepaddle>=1.6
- pgl>=1.0.0

## Hyperparameters
All the hyper parameters are saved in config.yaml file. So before training, you can open the config.yaml to modify the hyper parameters as you like.

for example, you can change the \"use_cuda\" to \"True \" in order to use GPU for training or modify \"data_path\" to specify the data you want.

Some important hyper parameters in config.yaml:
- **use_cuda**: use GPU to train model
- **data_path**: the directory of dataset that you want to load
- **lr**: learning rate
- **neg_num**: number of negative samples.
- **num_walks**: number of walks started from each node
- **walk_length**: walk length
- **metapath**: meta path scheme

## Metapath randomwalk sampling
Before training, we should generate some metapath random walks to train skipgram model. we can run the below command to produce metapath randomwalk data.
```sh
python sample.py -c config.yaml
```

## Training and Testing
After finishing metapath randomwalk sampling, you can run the below command to train and test the model.
```sh
python main.py -c config.yaml

python multi_class.py --dataset ./data/out_aminer_CPAPC/author_label.txt --word2id ./checkpoints/train.metapath2vec/word2id.pkl  --ckpt_path ./checkpoints/train.metapath2vec/model_epoch5/

```

## Experiment results
| train_percent | Metric   | PGL Result | Reported Result |
|---------------|----------|------------|-----------------|
| 50%           | macro-F1 | 0.9249     | 0.9314          |
| 50%           | micro-F1 | 0.9283     | 0.9365          |
