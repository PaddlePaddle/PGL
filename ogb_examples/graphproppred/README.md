# Graph Property Prediction for Open Graph Benchmark (OGB)

[The Open Graph Benchmark (OGB)](https://ogb.stanford.edu/) is a collection of benchmark datasets, data loaders, and evaluators for graph machine learning. Here we complete the Graph Property Prediction task based on PGL.

### Requirements

- paddlpaddle 1.7.1
- pgl 1.0.1
- ogb

NOTE: To install ogb that is fited for this project, run below command to install ogb
```
git clone https://github.com/snap-stanford/ogb.git
git checkout 482c40bc9f31fe25f9df5aa11c8fb657bd2b1621
python setup.py install
```

### How to run
For example, use GPU to train model on ogbg-molhiv dataset.
```
python main_pgl.py --use_cuda --dataset ogbg-molhiv
```
