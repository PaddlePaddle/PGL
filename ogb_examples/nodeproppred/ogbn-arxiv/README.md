# Graph Node Prediction for Open Graph Benchmark (OGB) Arxiv dataset

[The Open Graph Benchmark (OGB)](https://ogb.stanford.edu/) is a collection of benchmark datasets, data loaders, and evaluators for graph machine learning. Here we complete the Graph Node Prediction task based on PGL.


### Requirements

paddlpaddle >= 1.7.1

pgl 1.0.2

ogb 1.1.1


### How to Run

```
CUDA_VISIBLE_DEVICES=0 python train.py \
    --use_cuda 1 \
    --num_workers 4 \
    --output_path ./output/model_1 \
    --batch_size 1024 \
    --test_batch_size 512 \
    --epoch 100 \
    --learning_rate 0.001 \
    --full_batch 0 \
    --model gaan \
    --drop_rate 0.5 \
    --samples 8 8 8 \
    --test_samples 20 20 20 \
    --hidden_size 256
```
or

```
sh run.sh
```

The best record will be saved in ./output/model_1/best.txt.


### Hyperparameters
- use_cuda: whether to use gpu or not
- num_workers: the nums of sample workers
- output_path: path to save the model
- batch_size: batch size
- epoch: number of training epochs
- learning_rate: learning rate
- full_batch: run full batch of graph
- model: model to run, now gaan, sage, gcn, eta are available
- drop_rate: drop rate of the feature layers
- samples: the sample nums of each GNN layers
- hidden_size: the hidden size

### Performance
We train our models for 100 epochs and report the **acc** on the test dataset.
|dataset|mean|std|#experiments|
|-|-|-|-|
|ogbn-arxiv|0.7197|0.0024|16|
