# Unsupervised GraphSAGE in PGL
[GraphSAGE](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf) is a general inductive framework that leverages node feature
information (e.g., text attributes) to efficiently generate node embeddings for previously unseen data. Instead of training individual embeddings for each node, GraphSAGE learns a function that generates embeddings by sampling and aggregating features from a node’s local neighborhood. Based on PGL, we reproduce GraphSAGE algorithm and reach the same level of indicators as the paper in Reddit Dataset. Besides, this is an example of subgraph sampling and training in PGL.
For purpose of unsupervised learning, we use graph edges as positive samples for graphsage training.
### Datasets(Quickstart)
The dataset `./sample.txt` is handcrafted bigraph for quick demo purpose, which format is `src \t dst`.
### Dependencies
```txt
- paddlepaddle>=1.6
- pgl
```
### How to run
#### 1. Training
```sh
python train.py --data_path ./sample.txt --num_nodes 2000 --phase train
```
#### 2. Predicting
```sh
python train.py --data_path ./sample.txt --num_nodes 2000 --phase predict
```
The resulted node embedding is stored in `emb.npy` file, which latter can be loaded using `np.load`.
#### Hyperparameters
- epoch: Number of epochs default (1)
- use_cuda: Use gpu if assign use_cuda. 
- layer_type: We support 4 aggregator types including "graphsage_mean", "graphsage_maxpool", "graphsage_meanpool" and "graphsage_lstm".
- sample_workers: The number of workers for multiprocessing subgraph sample.
- lr: Learning rate.
- batch_size: Batch size.
- samples: The max neighbors sampling rate for each hop. (default: [10, 10])
- num_layers: The number of layer for graph sampling. (default: 2)
- hidden_size: The hidden size of the GraphSAGE models.
- checkpoint. Path for model checkpoint at each epoch. (default: 'model_ckpt')
