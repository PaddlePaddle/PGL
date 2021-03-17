# node2vec: Scalable Feature Learning for Networks 
[Node2vec](https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf) is an algorithmic framework for representational learning on graphs. Given any graph, it can learn continuous feature representations for the nodes, which can then be used for various downstream machine learning tasks. Based on PGL, we reproduce node2vec algorithms and reach the same level of indicators as the paper.
## Datasets
The datasets contain two networks: [BlogCatalog](http://socialcomputing.asu.edu/datasets/BlogCatalog3) and [Arxiv](http://snap.stanford.edu/data/ca-AstroPh.html). 
## Dependencies
- paddlepaddle>=1.4
- pgl

## How to run

For examples, use gpu to train gcn on cora dataset.
```sh
# multiclass task example
python node2vec.py --use_cuda --dataset BlogCatalog --save_path ./tmp/node2vec_BlogCatalog/ --offline_learning --epoch 400

python multi_class.py --use_cuda --ckpt_path ./tmp/node2vec_BlogCatalog/paddle_model --epoch 1000

# link prediction task example
python node2vec.py --use_cuda --dataset ArXiv --save_path
./tmp/node2vec_ArXiv --offline_learning --epoch 10

python link_predict.py --use_cuda --ckpt_path ./tmp/node2vec_ArXiv/paddle_model --epoch 400
```

## Hyperparameters
- dataset: The citation dataset "BlogCatalog" and "ArXiv".
- use_cuda: Use gpu if assign use_cuda. 

### Experiment results
Dataset|model|Task|Metric|PGL Result|Reported Result 
--|--|--|--|--|--
BlogCatalog|deepwalk|multi-label classification|MacroF1|0.250|0.211
BlogCatalog|node2vec|multi-label classification|MacroF1|0.262|0.258
ArXiv|deepwalk|link prediction|AUC|0.9538|0.9340
ArXiv|node2vec|link prediction|AUC|0.9541|0.9366
