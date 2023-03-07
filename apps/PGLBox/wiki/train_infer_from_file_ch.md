# 训练或预测指定文件目录

为了满足用户对于只训练部分节点，或者只预测部分节点的需求，PGLBox提供了相应的功能支持。

## 训练指定文件目录

在配置yaml文件时，配置项 train_start_nodes 用于指定训练的起始节点目录。如果直接置空，则默认会使用原先加载图数据的点来进行训练。
如果需要配置，简单举例如下：
``` shell
graph_data_local_path: /your/graph_data_path/
train_start_nodes: "train_nodes"
```
上述配置项意味着，目录结构如下:
``` shell
/your/graph_data_path/
|--user2item
|--node_types
|--train_nodes
```
其中，train_nodes文件夹中的格式需要与node_types保持一致，如果hadoop_shard功能为True的时候，也需要分片为1000个part。

## 预测指定文件目录

在配置yaml文件中，配置项 infer_nodes 用于指定 infer 的节点目录。如果直接置空，则默认会使用原先加载图数据的点来进行infer。
如果需要配置，简单举例如下：
``` shell
graph_data_local_path: /your/graph_data_path/
infer_nodes: "infer_nodes"
```
上述配置项意味着，目录结构如下：
``` shell
/your/graph_data_path/
|--user2item
|--node_types
|--infer_nodes
```
