# Building Graph Convolutional Network


[Graph Convolutional Network \(GCN\)](https://arxiv.org/abs/1609.02907) is a powerful neural network designed for machine learning on graphs. Based on PGL, we reproduce GCN algorithms and reach the same level of indicators as the paper in citation network benchmarks.

### Simple example to build GCN

To build a gcn layer, one can use our pre-defined ```pgl.layers.gcn``` or just write a gcn layer with message passing interface.
```python
import paddle.fluid as fluid
def gcn_layer(graph_wrapper, node_feature, hidden_size, act):
    def send_func(src_feat, dst_feat, edge_feat):
        return src_feat["h"]
    
    def recv_func(msg):
        return fluid.layers.sequence_pool(msg, "sum")
    
    message = graph_wrapper.send(send_func, nfeat_list=[("h", node_feature)])
    output = graph_wrapper.recv(recv_func, message)
    output = fluid.layers.fc(output, size=hidden_size, act=act)
    return output
```

### Datasets

The datasets contain three citation networks: CORA, PUBMED, CITESEER. The details for these three datasets can be found in the [paper](https://arxiv.org/abs/1609.02907).

### Dependencies

- paddlepaddle>=1.4 (The speed can be faster in 1.5.)
- pgl

### Performance

We train our models for 200 epochs and report the accuracy on the test dataset.

| Dataset | Accuracy | Speed with paddle 1.4 <br> (epoch time) | Speed with paddle 1.5 <br> (epoch time)|
| --- | --- | --- |---|
| Cora | ~81% | 0.0106s | 0.0104s | 
| Pubmed | ~79% | 0.0210s  | 0.0154s |
| Citeseer | ~71% | 0.0175s | 0.0177s | 


### How to run

For examples, use gpu to train gcn on cora dataset.
```
python train.py --dataset cora --use_cuda
```

#### Hyperparameters

- dataset: The citation dataset "cora", "citeseer", "pubmed".
- use_cuda: Use gpu if assign use_cuda. 


### View the Code

See the code [here](gcn_examples_code.html)
