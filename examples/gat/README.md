# GAT: Graph Attention Networks

[Graph Attention Networks \(GAT\)](https://arxiv.org/abs/1710.10903) is a novel architectures that operate on graph-structured data, which leverages masked self-attentional layers to address the shortcomings of prior methods based on graph convolutions or their approximations. Based on PGL, we reproduce GAT algorithms and reach the same level of indicators as the paper in citation network benchmarks.
### Simple example to build single head GAT

To build a gat layer,  one can use our pre-defined ```pgl.nn.GATConv``` or just write a gat layer with message passing interface.

```python

class CustomGATConv(nn.Layer):
    def __init__(self,
                 input_size, hidden_size,
                 ):

        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.linear = nn.Linear(input_size, hidden_size)
        self.weight_src = self.create_parameter(shape=[ hidden_size ])
        self.weight_dst = self.create_parameter(shape=[ hidden_size ])

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def send_attention(self, src_feat, dst_feat, edge_feat):
        alpha = src_feat["src"] + dst_feat["dst"]
        alpha = self.leaky_relu(alpha)
        return {"alpha": alpha, "h": src_feat["h"]}

    def reduce_attention(self, msg):
        alpha = msg.reduce_softmax(msg["alpha"])
        feature = msg["h"]
        feature = feature * alpha
        feature = msg.reduce(feature, pool_type="sum")
        return feature

    def forward(self, graph, feature):
        feature = self.linear(feature)
        attn_src = paddle.sum(feature * self.weight_src, axis=-1)
        attn_dst = paddle.sum(feature * self.weight_dst, axis=-1)
        msg = graph.send(
            self.send_attention,
            src_feat={"src": attn_src,
                      "h": feature},
            dst_feat={"dst": attn_dst})
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
