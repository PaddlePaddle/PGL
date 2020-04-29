# GAT: Graph Attention Networks

[Graph Attention Networks \(GAT\)](https://arxiv.org/abs/1710.10903) is a novel architectures that operate on graph-structured data, which leverages masked self-attentional layers to address the shortcomings of prior methods based on graph convolutions or their approximations. Based on PGL, we reproduce GAT algorithms and reach the same level of indicators as the paper in citation network benchmarks.
### Simple example to build single head GAT

To build a gat layer,  one can use our pre-defined ```pgl.layers.gat``` or just write a gat layer with message passing interface.
```python
import paddle.fluid as fluid
def gat_layer(graph_wrapper, node_feature, hidden_size):
    def send_func(src_feat, dst_feat, edge_feat):
        logits = src_feat["a1"] + dst_feat["a2"]
        logits = fluid.layers.leaky_relu(logits, alpha=0.2)
        return {"logits": logits, "h": src_feat }
    
    def recv_func(msg):
        norm = fluid.layers.sequence_softmax(msg["logits"])
        output = msg["h"] * norm
        return output
    
    h = fluid.layers.fc(node_feature, hidden_size, bias_attr=False, name="hidden")
    a1 = fluid.layers.fc(node_feature, 1, name="a1_weight")
    a2 = fluid.layers.fc(node_feature, 1, name="a2_weight")
    message = graph_wrapper.send(send_func,
            nfeat_list=[("h", h), ("a1", a1), ("a2", a2)])
    output = graph_wrapper.recv(recv_func, message)
    return output
```


### Datasets

The datasets contain three citation networks: CORA, PUBMED, CITESEER. The details for these three datasets can be found in the [paper](https://arxiv.org/abs/1609.02907).

### Dependencies

- paddlepaddle>=1.6
- pgl

### Performance

We train our models for 200 epochs and report the accuracy on the test dataset.

| Dataset | Accuracy |
| --- | --- |
| Cora | ~83% | 
| Pubmed | ~78% |
| Citeseer | ~70% | 

### How to run

For examples, use gpu to train gat on cora dataset.
```
python train.py --dataset cora --use_cuda
```

#### Hyperparameters

- dataset: The citation dataset "cora", "citeseer", "pubmed".
- use_cuda: Use gpu if assign use_cuda. 
