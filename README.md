# Paddle Graph Learning (PGL) 

[DOC](https://pgl.readthedocs.io/en/latest/) | [Quick Start](https://pgl.readthedocs.io/en/latest/instruction.html) | [中文](./README.zh.md)

Paddle Graph Learning (PGL) is an efficient and flexible graph learning framework based on [PaddlePaddle](https://github.com/PaddlePaddle/Paddle). 


<img src="./docs/source/_static/framework_of_pgl.png" alt="The Framework of Paddle Graph Learning (PGL)" width="800">


We provide python interfaces for storing/reading/querying graph structured data and two fundamental computational interfaces, which are walk based paradigm and message-passing based paradigm as shown in the above framework of PGL, for building cutting-edge graph learning algorithms.  Combined with the PaddlePaddle deep learning framework, we are able to support both graph representation learning models and graph neural networks, and thus our framework has a wide range of graph-based applications.


## Highlight: Efficient and Flexible Message Passing Paradigm

One of the most important benefits of graph neural networks compared to other models is the ability to use node-to-node connectivity information, but coding the communication between nodes is very cumbersome. At PGL we adopt **Message Passing Paradigm** similar to [DGL](https://github.com/dmlc/dgl) to help to build a customize graph neural network easily. Users only need to write ```send``` and ```recv``` functions to easily implement a simple GCN. As shown in the following figure, for the first step the send function is defined on the edges of the graph, and the user can customize the send function ![](http://latex.codecogs.com/gif.latex?\\phi^e}) to send the message from the source to the target node. For the second step, the recv function ![](http://latex.codecogs.com/gif.latex?\\phi^v}) is responsible for aggregating ![](http://latex.codecogs.com/gif.latex?\\oplus}) messages together from different sources.



<img src="./docs/source/_static/message_passing_paradigm.png" alt="The basic idea of message passing paradigm" width="800">

As shown in the left of the following figure, to adapt general user-defined message aggregate functions, DGL uses the degree bucketing method to combine nodes with the same degree into a batch and then apply an aggregate function ![](http://latex.codecogs.com/gif.latex?\\oplus}) on each batch serially. For our PGL UDF aggregate function, we organize the message as a [LodTensor](http://www.paddlepaddle.org/documentation/docs/en/1.4/user_guides/howto/basic_concept/lod_tensor_en.html) in [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) taking the message as variable length sequences. And we **utilize the features of LodTensor in Paddle to obtain fast parallel aggregation**. 


<img src="./docs/source/_static/parallel_degree_bucketing.png" alt="The parallel degree bucketing of PGL" width="800">


Users only need to call the ```sequence_ops``` functions provided by Paddle to easily implement efficient message aggregation. For examples, using ```sequence_pool``` to sum the neighbor message.
```python
    import paddle.fluid as fluid
    def recv(msg):
        return fluid.layers.sequence_pool(msg, "sum")
```


Although DGL does some kernel fusion optimization for general sum, max and other aggregate functions with scatter-gather. For **complex user-defined functions** with degree bucketing algorithm, the serial execution for each degree bucket cannot take full advantage of the performance improvement provided by GPU. However, operations on the PGL LodTensor-based message is performed in parallel, which can fully utilize GPU parallel optimization. In our experiments, PGL can reach up to 13 times the speed of DGL with complex user-defined functions. Even without scatter-gather optimization, PGL still has excellent performance. Of course, we still provide build-in scatter-optimized message aggregation functions.

## Performance


We test all the GNN algorithms with Tesla V100-SXM2-16G running for 200 epochs to get average speeds. And we report the accuracy on test dataset without early stoppping.

| Dataset | Model |  PGL Accuracy | PGL speed (epoch time) | DGL 0.3.0 speed (epoch time) |
| -------- | ----- | ----------------- | ------------ | ------------------------------------ |
| Cora | GCN |81.75% | 0.0047s | **0.0045s** |
| Cora | GAT | 83.5% | **0.0119s** | 0.0141s |
| Pubmed | GCN |79.2% |**0.0049s** |0.0051s |
| Pubmed | GAT | 77% |0.0193s|**0.0144s**|
| Citeseer | GCN |70.2%| **0.0045** |0.0046s|
| Citeseer | GAT |68.8%| **0.0124s** |0.0139s|

If we use complex user-defined aggregation like [GraphSAGE-LSTM](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf) that aggregates neighbor features with LSTM ignoring the order of recieved messages, the optimized message-passing in DGL will be forced to degenerate into degree bucketing scheme. The speed performance will be much slower than the one implemented in PGL. Performances may be various with different scale of the graph, in our experiments, PGL can reach up to 13 times the speed of DGL.

| Dataset |   PGL speed (epoch time) | DGL 0.3.0 speed (epoch time) | Speed up|
| -------- |  ------------ | ------------------------------------ |----|
| Cora | **0.0186s** | 0.1638s | 8.80x|
| Pubmed | **0.0388s** |0.5275s | 13.59x|
| Citeseer | **0.0150s** | 0.1278s | 8.52x |



## System requirements

PGL requires:

* paddle >= 1.6
* cython


PGL supports both Python 2 & 3


## Installation

The current version of PGL is 1.0.0. You can simply install it via pip.

```sh
pip install pgl
```

## The Team

PGL is developed and maintained by NLP and Paddle Teams at Baidu

## License

PGL uses Apache License 2.0.
