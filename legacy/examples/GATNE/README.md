# GATNE: General Attributed Multiplex HeTerogeneous Network Embedding
[GATNE](https://arxiv.org/pdf/1905.01669.pdf) is a algorithms framework for embedding large-scale Attributed Multiplex Heterogeneous Networks(AMHN). Given a heterogeneous graph, which consists of nodes and edges of multiple types, it can learn continuous feature representations for every node. Based on PGL, we reproduce GATNE algorithm. 

## Datasets
YouTube dataset contains 2000 nodes, 1310617 edges and 5 edge types. And we use YouTube dataset for example.

You can dowload YouTube datasets from [here](https://github.com/THUDM/GATNE/tree/master/data)

After downloading the data, put them, let's say, in ./data/ . Note that the current directory is the root directory of GATNE model. Then in ./data/youtube/ directory, there are three files:
* train.txt
* valid.txt
* test.txt

Then you can run the below command to preprocess the data.
```sh
python data_process.py --input_file ./data/youtube/train.txt --output_file ./data/youtube/nodes.txt
```

## Dependencies
- paddlepaddle>=1.6
- pgl>=1.0.0

## Hyperparameters
All the hyper parameters are saved in config.yaml file. So before training GATNE model, you can open the config.yaml to modify the hyper parameters as you like.

for example, you can change the \"use_cuda\" to \"True \" in order to use GPU for training or modify \"data_path\" to use different dataset.

Some important hyper parameters in config.yaml:
- use_cuda: use GPU to train model
- data_path: the directory of dataset
- lr: learning rate
- neg_num: number of negatie samples.
- num_walks: number of walks started from each node
- walk_length: walk length

## How to run
Then run the below command:
```sh
python main.py -c config.yaml
```

### Experiment results
|     | PGL result | Reported result |
|:---:|------------|-----------------|
| AUC | 84.83     | 84.61           |
| PR  | 82.77     | 81.93           |
| F1  | 76.98     | 76.83           |
