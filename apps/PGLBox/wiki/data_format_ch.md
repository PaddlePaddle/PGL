# 图数据格式

本文主要介绍使用PGLBox进行图表示学习训练时所需要的图数据格式。

假设我们有author、paper和inst三种类型的节点。 边有author2paper（author - write - paper）、author2inst（author - belong to - inst）和paper2paper（paper - cite - paper）三种类型的边。

那么我们需要准备节点类型文件和边关系文件。

## 节点类型文件

节点类型文件的格式为：

```
node_type \t node_id
```

其中，node_type是节点的类型，例如为paper、author或inst。而node_id为**uint64**的数字，**但注意不能为0**。

```bash

# 简单举例

inst	523515
inst	614613
inst	611434
paper	2342353241
paper	2451413511
author	9905123492
author	9845194235

```

## 带slot特征的节点类型文件

如果节点带有离散slot特征，则节点类型文件的格式为：

```bash

node_type \t node_id \t slot1:slot1_value \t slot2:slot2_value \t slot3:slot3_value

```

然后在`./user_configs/`目录的配置文件中，修改slots参数为：

```bash
slots: ["slot1", "slot2", "slot3"]

```

注意这些slot都是数字。不同的slot表示不同的特征类型。比如”11“表示性别，”21“表示年龄等。

例如，author类型节点有性别和年龄两种slot特征，分别用“11”和“21”来表示。paper类型节点有发表年份这种slot特征，用“101”来表示。那么节点类型文件可以表示为：

```bash
# 简单举例

inst	523515
inst	614613
inst	611434
paper	2342353241    101:2020
paper	2451413511    101:2019
author	9905123492    11:1     21:30
author	9845194235    11:0     21:24

```

在配置文件中，slots这个参数则设置为：

```bash
slots: ["11", "21", "101"]
```

## 边关系文件

每一种边类型都需要单独一个子目录来保存边关系。例如这里有三种边类型，所以需要三个子目录来保存边关系。

边关系文件的格式为：

```
src_node_id \t dst_node_id

```
其中，这些src_node_id和dst_node_id均为**uint64**的数字。

```

# 简单举例: author2paper

9905123492	2451413511
9845194235	2342353241
9845194235	2451413511 

```

## 数据目录结构

根据上述的数据格式进行数据准备，我们可以得到如下四个数据子目录：

```bash
/your/path/to/graph_data
    |
    |--node_types/
    |--author2paper/
    |--paper2paper/
    |--author2inst/
```

其中， `node_types`子目录保存的就是节点类型以及相应的节点slot特征（如有）。`author2paper`子目录保存的是（author - write - paper）类型的边，`paper2paper`子目录保存的是（paper - cite - paper）类型的边，`author2inst`子目录保存的是（author - belong to - inst）类型的边。

每个子目录下面都有很多个文件，保存相应的内容（之所以是子目录的形式，是考虑到一般情况下，大规模数据都会分片保存在多个文件里面）。


## 图数据分片

为了能够加快图数据的加载速度，我们需要对图数据进行分片，分片之后图引擎就可以多线程并行加载，大大提高加载图数据的速度。
因此，可以使用我们提供的sharding_tool工具，将节点类型文件和边类型文件进行分片。
详细使用方法请参考[这里](https://github.com/PaddlePaddle/PGL/tree/main/apps/PGLBox/sharding_tool/)。


## 配置文件设置

对图数据进行分片处理后，我们假设处理好的图数据保存在这个目录下`/your/path/to/preprocessed_graph_data`，即：

```bash
/your/path/to/preprocessed_graph_data
    |
    |--node_types/
    |--author2paper/
    |--paper2paper/
    |--author2inst/
```

那么接下来我们只需要在配置文件中设定以下几个参数，就可以换成这份处理好的图数据来训练了：

```bash
graph_data_local_path: "/your/path/to/preprocessed_graph_data"

etype2files: "author2paper:author2paper,paper2paper:paper2paper,author2inst:author2inst"

ntype2files: "author:node_types,paper:node_types,inst:node_types"

hadoop_shard: True

num_part: 1000

symmetry: True
```

**注意**: 一般我们是用docker容器来训练模型，那么要注意`graph_data_local_path`在docker容器里面的路径映射关系。不然会因为数据路径错误导致读取不到图数据。
