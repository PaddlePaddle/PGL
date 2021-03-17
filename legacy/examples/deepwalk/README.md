# Distributed Deepwalk in PGL
[Deepwalk](https://arxiv.org/pdf/1403.6652.pdf) is an algorithmic framework for representational learning on graphs. Given any graph, it can learn continuous feature representations for the nodes, which can then be used for various downstream machine learning tasks. Based on PGL, we reproduce distributed deepwalk algorithms and reach the same level of indicators as the paper.

## Datasets
The datasets contain two networks: [BlogCatalog](http://socialcomputing.asu.edu/datasets/BlogCatalog3) and [Arxiv](http://snap.stanford.edu/data/ca-AstroPh.html). 
## Dependencies
- paddlepaddle>=1.6
- pgl

## How to run

For examples, use gpu to train deepwalh on BlogCatalog and ArXiv dataset.
```sh
# multiclass task example
python deepwalk.py --use_cuda --dataset BlogCatalog --save_path ./tmp/deepwalk_BlogCatalog/ --offline_learning --epoch 400

python multi_class.py --use_cuda --ckpt_path ./tmp/deepwalk_BlogCatalog/paddle_model --epoch 1000

# link prediction task example
python deepwalk.py --use_cuda --dataset ArXiv --save_path
./tmp/deepwalk_ArXiv --offline_learning --epoch 100

python link_predict.py --use_cuda --ckpt_path ./tmp/deepwalk_ArXiv/paddle_model --epoch 400
```

## Hyperparameters
- dataset: The citation dataset "BlogCatalog" and "ArXiv".
- use_cuda: Use gpu if assign use_cuda. 

### Experiment results
Dataset|model|Task|Metric|PGL Result|Reported Result 
--|--|--|--|--|--
BlogCatalog|deepwalk|multi-label classification|MacroF1|0.250|0.211
ArXiv|deepwalk|link prediction|AUC|0.9538|0.9340
