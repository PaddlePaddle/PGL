# Graph Link Prediction for Open Graph Benchmark (OGB) PPA dataset

[The Open Graph Benchmark (OGB)](https://ogb.stanford.edu/) is a collection of benchmark datasets, data loaders, and evaluators for graph machine learning. Here we complete the Graph Link Prediction task based on PGL.


### Requirements

paddlpaddle >= 1.7.1

pgl 1.0.2

ogb


### How to Run

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --use_cuda 1  --num_workers 4 --output_path ./output/model_1 --batch_size 65536 --epoch 1000 --learning_rate 0.005  --hidden_size 256 
```

The best record will be saved in ./output/model_1/best.txt.
