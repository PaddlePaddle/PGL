# GATv2: How Attentive are Graph Attention Networks?

[How Attentive are Graph Attention Networks? \(GATv2\)](https://arxiv.org/abs/2105.14491) is a dynamic graph attention variant after modifying the order of operations of GAT. Based on PGL, we reproduce GATv2 algorithms and reach the same level of indicators as the paper in citation network benchmarks.

### Simple example to build single head GATv2

To build a gatv2 layer,  one can use our pre-defined ```pgl.nn.GATv2Conv``` or just write a gatv2 layer with message passing interface.

```python

class CustomGATv2Conv(nn.Layer):
    def __init__(self, input_size, hidden_size):
        super(CustomGATv2Conv, self).__init__()

        self.hidden_size = hidden_size

        self.linear = nn.Linear(input_size, hidden_size)
        self.attn = self.create_parameter(shape=[hidden_size])
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def send_attention(self, src_feat, dst_feat, edge_feat):
        alpha = src_feat["src"] + dst_feat["dst"]
        alpha = self.leaky_relu(alpha)
        alpha = paddle.sum(alpha * self.attn, axis=-1)
        return {"alpha": alpha, "h": src_feat["src"]}

    def reduce_attention(self, msg):
        alpha = msg.reduce_softmax(msg["alpha"])
        alpha = paddle.reshape(alpha, [-1, 1])
        feature = msg["h"]
        feature = feature * alpha
        feature = msg.reduce(feature, pool_type="sum")
        return feature

    def forward(self, graph, feature):
        feature = self.linear(feature)
        msg = graph.send(
            self.send_attention, 
            src_feat={"src": feature}, 
            dst_feat={"dst": feature}
        )
        output = graph.recv(reduce_func=self.reduce_attention, msg=msg)
        return output
```


### Datasets

The datasets contain three citation networks: CORA, PUBMED, CITESEER. The details for these three datasets can be found in the [paper](https://arxiv.org/abs/1609.02907).

### Dependencies

- paddlepaddle==2.0.0
- pgl==2.1

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
python train.py --dataset cora
```

#### Hyperparameters

- dataset: The citation dataset "cora", "citeseer", "pubmed".
- use_cuda: Use gpu if assign use_cuda. 
