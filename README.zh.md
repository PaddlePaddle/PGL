# Paddle Graph Learning (PGL) 

[文档](https://pgl.readthedocs.io/en/latest/) | [快速开始](https://pgl.readthedocs.io/en/latest/instruction.html) | [English](./README.md)

Paddle Graph Learning (PGL)是一个基于[PaddlePaddle](https://github.com/PaddlePaddle/Paddle)的高效易用的图学习框架

<img src="./docs/source/_static/framework_of_pgl.png" alt="The Framework of Paddle Graph Learning (PGL)" width="800">

我们提供一系列的Python接口用于存储/读取/查询图数据结构，并且提供基于游走（Walk Based）以及消息传递（Message Passing）两种计算范式的计算接口（见上图）。利用这些接口，我们就能够轻松的搭建最前沿的图学习算法。结合PaddlePaddle深度学习框架，我们的框架基本能够覆盖大部分的图网络应用，包括图表示学习以及图神经网络。


## 特色：高效灵活的消息传递范式


对比于一般的模型，图神经网络模型最大的优势在于它利用了节点与节点之间连接的信息。但是，如何通过代码来实现建模这些节点连接十分的麻烦。PGL采用与[DGL](https://github.com/dmlc/dgl)相似的**消息传递范式**用于作为构建图神经网络的接口。用于只需要简单的编写```send```还有```recv```函数就能够轻松的实现一个简单的GCN网络。如下图所示，首先，send函数被定义在节点之间的边上，用户自定义send函数![](http://latex.codecogs.com/gif.latex?\\phi^e})会把消息从源点发送到目标节点。然后，recv函数![](http://latex.codecogs.com/gif.latex?\\phi^v})负责将这些消息用汇聚函数 ![](http://latex.codecogs.com/gif.latex?\\oplus}) 汇聚起来。

<img src="./docs/source/_static/message_passing_paradigm.png" alt="The basic idea of message passing paradigm" width="800">

如下面左图所示，为了去适配用户定义的汇聚函数，DGL使用了Degree Bucketing来将相同度的节点组合在一个块，然后将汇聚函数![](http://latex.codecogs.com/gif.latex?\\oplus})作用在每个块之上。而对于PGL的用户定义汇聚函数，我们则将消息以PaddlePaddle的[LodTensor](http://www.paddlepaddle.org/documentation/docs/en/1.4/user_guides/howto/basic_concept/lod_tensor_en.html)的形式处理，将若干消息看作一组变长的序列，然后利用**LodTensor在PaddlePaddle的特性进行快速平行的消息聚合**。

<img src="./docs/source/_static/parallel_degree_bucketing.png" alt="The parallel degree bucketing of PGL" width="800">

用户只需要简单的调用PaddlePaddle序列相关的函数```sequence_ops```就可以实现高效的消息聚合了。举个例子，下面就是简单的利用```sequence_pool```来做邻居消息求和。

```python
    import paddle.fluid as fluid
    def recv(msg):
        return fluid.layers.sequence_pool(msg, "sum")
```

尽管DGL用了一些内核融合（kernel fusion）的方法来将常用的sum，max等聚合函数用scatter-gather进行优化。但是对于**复杂的用户定义函数**，他们使用的Degree Bucketing算法，仅仅使用串行的方案来处理不同的分块，并不同充分利用GPU进行加速。然而，在PGL中我们使用基于LodTensor的消息传递能够充分地利用GPU的并行优化，在复杂的用户定义函数下，PGL的速度在我们的实验中甚至能够达到DGL的13倍。即使不使用scatter-gather的优化，PGL仍然有高效的性能表现。当然，我们也是提供了scatter优化的聚合函数。


## 性能测试


我们用Tesla V100-SXM2-16G测试了下列所有的GNN算法，每一个算法跑了200个Epoch来计算平均速度。准确率是在测试集上计算出来的，并且我们没有使用Early-stopping策略。

| 数据集 | 模型 |  PGL准确率 | PGL速度 (epoch) | DGL 0.3.0 速度 (epoch) |
| -------- | ----- | ----------------- | ------------ | ------------------------------------ |
| Cora | GCN |81.75% | 0.0047s | **0.0045s** |
| Cora | GAT | 83.5% | **0.0119s** | 0.0141s |
| Pubmed | GCN |79.2% |**0.0049s** |0.0051s |
| Pubmed | GAT | 77% |0.0193s|**0.0144s**|
| Citeseer | GCN |70.2%| **0.0045** |0.0046s|
| Citeseer | GAT |68.8%| **0.0124s** |0.0139s|

如果我们使用复杂的用户定义聚合函数，例如像[GraphSAGE-LSTM](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf)那样忽略邻居信息的获取顺序，利用LSTM来聚合节点的邻居特征。DGL所使用的消息传递函数将退化成Degree Bucketing模式，在这个情况下DGL实现的模型会比PGL的慢的多。模型的性能会随着图规模而变化，在我们的实验中，PGL的速度甚至能够能达到DGL的13倍。

| 数据集 |   PGL速度 (epoch) | DGL 0.3.0 速度 (epoch time) | 加速比 |
| -------- |  ------------ | ------------------------------------ |----|
| Cora | **0.0186s** | 0.1638s | 8.80x|
| Pubmed | **0.0388s** |0.5275s | 13.59x|
| Citeseer | **0.0150s** | 0.1278s | 8.52x |


## 依赖

PGL依赖于:

* paddle >= 1.5
* networkx
* cython


PGL支持Python 2和3。


## 安装

当前，PGL的版本是0.1.0.beta。你可以简单的用pip进行安装。

```sh
pip install pgl
```

## 团队

PGL由百度的NLP以及Paddle团队共同开发以及维护。

## License

PGL uses Apache License 2.0.

