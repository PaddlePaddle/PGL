<img src="./docs/source/_static/logo.png" alt="The logo of Paddle Graph Learning (PGL)" width="320">

[![PyPi Latest Release](https://img.shields.io/pypi/v/pgl.svg)](https://pypi.org/project/pgl/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)

[文档](https://pgl.readthedocs.io/en/latest/) | [快速开始](https://pgl.readthedocs.io/en/latest/quick_start/instruction.html) | [English](./README.md)

## 最新消息

PGL v1.2 2020.11.20

- PGL团队提出统一的消息传递模型**UniMP**，刷新OGB**三**项榜单SOTA。你可以在[这里](./ogb_examples/nodeproppred/unimp)看到详细的代码。

- PGL团队提出基于**ERNIEsage**的二阶段召回与排序模型, 在COLING协办的[TextGraphs2020](https://competitions.codalab.org/competitions/23615)比赛中取得**第一名**。

- PGL团队倾力开发了**图神经网络公开课**,带你七天高效入门图神经网络。课程详情请参考[这里](https://aistudio.baidu.com/aistudio/course/introduce/1956)。

PGL v1.1 2020.4.29

- **ERNIESage**是PGL团队最新提出的模型，可以用于建模文本以及图结构信息。你可以在[这里](./examples/erniesage)看到详细的介绍。

- PGL现在提供[Open Graph Benchmark](https://github.com/snap-stanford/ogb)的一些例子，你可以在[这里](./ogb_examples)找到。

- 新增了图级别的算子包括**GraphPooling**以及[**GraphNormalization**](https://arxiv.org/abs/2003.00982)，这样你就能实现更多复杂的图级别分类模型。

- 新增PGL-KE工具包，里面包含许多经典知识图谱图嵌入算法，包括TransE, TransR, RotatE，详情可见[这里](./examples/pgl-ke)

------

Paddle Graph Learning (PGL)是一个基于[PaddlePaddle](https://github.com/PaddlePaddle/Paddle)的高效易用的图学习框架

<img src="./docs/source/_static/framework_of_pgl.png" alt="The Framework of Paddle Graph Learning (PGL)" width="800">

在最新发布的PGL中引入了异构图的支持，新增MetaPath采样支持异构图表示学习，新增异构图Message Passing机制支持基于消息传递的异构图算法，利用新增的异构图接口，能轻松搭建前沿的异构图学习算法。而且，在最新发布的PGL中，同时也增加了分布式图存储以及一些分布式图学习训练算法，例如，分布式deep walk和分布式graphsage。结合PaddlePaddle深度学习框架，我们的框架基本能够覆盖大部分的图网络应用，包括图表示学习以及图神经网络。


# 特色：高效性——支持Scatter-Gather及LodTensor消息传递


对比于一般的模型，图神经网络模型最大的优势在于它利用了节点与节点之间连接的信息。但是，如何通过代码来实现建模这些节点连接十分的麻烦。PGL采用与[DGL](https://github.com/dmlc/dgl)相似的**消息传递范式**用于作为构建图神经网络的接口。用于只需要简单的编写```send```还有```recv```函数就能够轻松的实现一个简单的GCN网络。如下图所示，首先，send函数被定义在节点之间的边上，用户自定义send函数![](http://latex.codecogs.com/gif.latex?\\phi^e)会把消息从源点发送到目标节点。然后，recv函数![](http://latex.codecogs.com/gif.latex?\\phi^v)负责将这些消息用汇聚函数 ![](http://latex.codecogs.com/gif.latex?\\oplus) 汇聚起来。

<img src="./docs/source/_static/message_passing_paradigm.png" alt="The basic idea of message passing paradigm" width="800">

如下面左图所示，为了去适配用户定义的汇聚函数，DGL使用了Degree Bucketing来将相同度的节点组合在一个块，然后将汇聚函数![](http://latex.codecogs.com/gif.latex?\\oplus)作用在每个块之上。而对于PGL的用户定义汇聚函数，我们则将消息以PaddlePaddle的[LodTensor](http://www.paddlepaddle.org/documentation/docs/en/1.4/user_guides/howto/basic_concept/lod_tensor_en.html)的形式处理，将若干消息看作一组变长的序列，然后利用**LodTensor在PaddlePaddle的特性进行快速平行的消息聚合**。

<img src="./docs/source/_static/parallel_degree_bucketing.png" alt="The parallel degree bucketing of PGL" width="800">

用户只需要简单的调用PaddlePaddle序列相关的函数```sequence_ops```就可以实现高效的消息聚合了。举个例子，下面就是简单的利用```sequence_pool```来做邻居消息求和。

```python
    import paddle.fluid as fluid
    def recv(msg):
        return fluid.layers.sequence_pool(msg, "sum")
```

尽管DGL用了一些内核融合（kernel fusion）的方法来将常用的sum，max等聚合函数用scatter-gather进行优化。但是对于**复杂的用户定义函数**，他们使用的Degree Bucketing算法，仅仅使用串行的方案来处理不同的分块，并不会充分利用GPU进行加速。然而，在PGL中我们使用基于LodTensor的消息传递能够充分地利用GPU的并行优化，在复杂的用户定义函数下，PGL的速度在我们的实验中甚至能够达到DGL的13倍。即使不使用scatter-gather的优化，PGL仍然有高效的性能表现。当然，我们也是提供了scatter优化的聚合函数。


### 性能测试
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


## 特色：易用性——原生支持异构图

图可以很方便的表示真实世界中事物之间的联系，但是事物的类别以及事物之间的联系多种多样，因此，在异构图中，我们需要对图网络中的节点类型以及边类型进行区分。PGL针对异构图包含多种节点类型和多种边类型的特点进行建模，可以描述不同类型之间的复杂联系。

### 支持异构图MetaPath walk采样
<img src="./docs/source/_static/metapath_sampling.png" alt="The metapath sampling in heterogeneous graph" width="800">
上图左边描述的是一个购物的社交网络，上面的节点有用户和商品两大类，关系有用户和用户之间的关系，用户和商品之间的关系以及商品和商品之间的关系。上图的右边是一个简单的MetaPath采样过程，输入metapath为UPU（user-product-user），采出结果为
<img src="./docs/source/_static/metapath_result.png" alt="The metapath result" width="320">
然后在此基础上引入word2vec等方法，支持异构图表示学习metapath2vec等算法。

### 支持异构图Message Passing机制

<img src="./docs/source/_static/him_message_passing.png" alt="The message passing mechanism on heterogeneous graph" width="800">
在异构图上由于节点类型不同，消息传递也方式也有所不同。如上图左边，它有五个邻居节点，属于两种不同的节点类型。如上图右边，在消息传递的时候需要把属于不同类型的节点分开聚合，然后在合并成最终的消息，从而更新目标节点。在此基础上PGL支持基于消息传递的异构图算法，如GATNE等算法。


## 特色：规模性——支持分布式图存储以及分布式学习算法

在大规模的图网络学习中，通常需要多机图存储以及多机分布式训练。如下图所示，PGL提供一套大规模训练的解决方案，我们利用[PaddleFleet](https://github.com/PaddlePaddle/Fleet)(支持大规模分布式Embedding学习)作为我们参数服务器模块以及一套简易的分布式存储方案，可以轻松在MPI集群上搭建分布式大规模图学习方法。

<img src="./docs/source/_static/distributed_frame.png" alt="The distributed frame of PGL" width="800">


## 丰富性——覆盖业界大部分图学习网络

下列是框架中部分已经实现的图网络模型，更多的模型在[这里](./examples)可以找到。详情请参考[这里](https://pgl.readthedocs.io/en/latest/introduction.html#highlight-tons-of-models)

| 模型 | 特点 |
|---|---|
| [**ERNIESage**](./examples/erniesage/) | 能同时建模文本以及图结构的ERNIE SAmple aggreGatE |
| GCN | 图卷积网络 |
| GAT | 基于Attention的图卷积网络 |
| GraphSage | 基于邻居采样的大规模图卷积网络 |
| unSup-GraphSage | 无监督学习的GraphSAGE |  
| LINE | 基于一阶、二阶邻居的表示学习 |  
| DeepWalk | DFS随机游走的表示学习 |  
| MetaPath2Vec | 基于metapath的表示学习 |
| Node2Vec | 结合DFS及BFS的表示学习 | 
| Struct2Vec | 基于结构相似的表示学习 |
| SGC | 简化的图卷积网络 | 
| GES | 加入节点特征的图表示学习方法 | 
| DGI | 基于图卷积网络的无监督表示学习 |
| GATNE | 基于MessagePassing的异构图表示学习 |

上述模型包含图表示学习，图神经网络以及异构图三部分，而异构图里面也分图表示学习和图神经网络。


## 依赖

PGL依赖于:

* paddle >= 1.6
* cython


PGL支持Python 2和3。


## 安装

你可以简单的用pip进行安装。

```sh
pip install pgl
```

## 团队

PGL由百度的NLP以及Paddle团队共同开发以及维护。

联系方式 E-mail: nlp-gnn[at]baidu.com

## License

PGL uses Apache License 2.0.
