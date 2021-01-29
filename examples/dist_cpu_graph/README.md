# MPI分布式图引擎

## Step 1: 目录结构说明

主要的配置文件都在`./config.yaml`内。

# Step 2: 数据格式
假如节点类型有u, t, f三种类型的节点. 边有u2t(u – t),  u2f(u – f), t2f(t – f) 三种边。那么需要构建如下四个文件，其中， 三个边的文件（u2t,  u2f,  t2f），一个节点类型文件(node_types)。

边文件的格式为:
```
src_node_id \t dst_node_id
```

节点类型文件格式为:
```
node_type \t node_id
```

其中, `node_id` 是一个**uint64**的数字, **但注意不能是0.**


## Step 3: 配置文件
所有的配置都可以在`config.yaml`文件中配置. 主要配置分为以下几个部分:

### hadoop 配置: 主要是训练数据的存放位置, 模型的输出位置
* `fs_name`: hadoop集群地址
  
* `fs_ugi`: hadoop集群ugi

* `graph_data_hdfs_path`: 训练数据在hadoop上的保存目录.训练数据的格式参考step 2.

其它具体的参数可以参考config.yaml文件.


## step 4: 运行

### Mpi 运行

上述配置完成后, 可以在项目的根目录运行下面的命令提交集群训练:
```
sh run.sh
```

## Step 5: 其它功能


### 1, 如何在PaddleCloud上查看日志?

打开k8s的某个节点页面后, 日志在`trainer_{num}.log`文件 其中, `num`是奇数则是trainer的log, 偶数是pserver的log.

### 2, 加入节点特征(side info)

如果有节点特征的话, 可以在node_types.txt文件中添加. 也就是将node_types.txt文件格式改为:

```
node_type \t node_id \t slot1:slot1_value \t slot2:slot2_value \t slot3:slot3_value
```


### 4, 分布式图存储与采样

考虑到有些图数据量很大, 单机保存整个图内存吃力. 因此设计了一个分布式图引擎, 让每台机器只保存部分的边和节点, 缓解单机内存吃力的问题, 同时也加速了图的加载过程. 这样就不需要进行"多进程共享内存"的模式了.

为了让每台机只保存部分节点和边, 在训练之前, 需要用hadoop对图数据进行分片. 数据分片的hadoop脚本在`./hadoop`目录.

在进入`./hadoop`目录对数据进行分片之前, 需要对`./config.yaml`中的两个参数进行如下设置: 

* `shard`: True

* `shard_num`: 100   # shard_num 就是分片的数量, 这个可以根据自己的需求进行设置, 但注意这里要设置一个大于训练节点数, 最好是整除的关系.


接下来进入`./hadoop`目录, 这里需要对`hadoop_run.sh`中的一些参数进行配置:

* `hadoop_bin`:  # 设置本地的hadoop

* `hadoop_base_output_path`: # 这是分片后的数据输出目录, 因此, 后面进行训练时, 需要设置`./config.yaml`中的`graph_data_hdfs_path` 与这个参数一致.

* `hadoop_intput_path`: # 这个是边的输入目录, 每种类型都需要分片一次.

* `hadoop_output_path`: # 分片后的边的输出目录, 这里要注意最后一级目录的名称需要跟`../config.yaml` 文件中的`edges_files`参数的`edge_file_or_directory`保持一致

* `hadoop_node_type_path`:  # node_types 文件所在目录

* `hadoop_node_type_output_path`  # node_types 分片后的输出目录, 注意最后一级目录要与`../config.yaml`中的`node_types_file`参数保持一致.

设置完成后, 运行下面的命令, 即可对数据进行分片: 

```
cd ./hadoop

sh hadoop_run.sh
```

### 5, 同构图的使用方法(deepwalk)

虽然这里是metapath2vec, 但对于同构图来说也是可以跑的. 例如只有user(u)节点的话, 只需要设置以下几个参数:

* `edge_files`:  "u2u: u2u_edges"

* `neg_sample_type`:  "average"

* `walk_mode`: "m2v"

* `metapath`: "u2u-u2u"

* `first_node_type`: "u"



## Step 6: 常见问题

### 1, UserWarning: [Errno 13] Permission denied.

```
/home/disk1/normandy/maybach/app-user-20200909102859-12341/workspace/python_gcc482/lib/python2.7/site-packages/sklearn/externals/joblib/_multiprocessing_helpers.py:38: UserWarning: [Errno 13] Permission denied.  joblib will operate in serial mode
```

如果有上述的Erron出现是正常的. 不是错误. 如果没有其它的log打印出来, 有可能是数据量很少.还不足以打印.


## Step 7: 本地Client使用

### 1. 环境要求：本地客户端可以是任意Python 以及一下环境
```
protobuf == 3.14.0
grpcio == 3.14.0
pgl == 1.2.0
```

### 2. 使用方法

将mpi运行加载图之后的server_endpoints(在日志 ./server_endpoints里)拷贝下来，放在server_endpoints文件里，即可使用.

```
from client.graph_client import DistCPUGraphClient

g = DistCPUGraphClient("server_endpoints", shard_num=1000)

```

## 主要Client API介绍

### 1. 遍历图上节点 `node_batch_iter`

参数：

`batch_size`: 每次返回节点个数

`node_type`: 节点类型 

`shuffle`: 乱序

`rank` 和 `nrank`： 控制sharding


返回: 迭代器

```

for batch_node in g.node_batch_iter(batch_size,
                            node_type=n_type,
                            shuffle=True,
                            rank=rank,
                            nrank=nrank):
    pass
```

### 2.采样邻居  `sample_predecessor` 和 `sample_successor`

参数：

`nodes`: 采样的节点

`max_degree`: 采样的最大邻居数

`edge_type`: 按照对应的边类型采样


返回：list of list，每一行为对应节点的邻居。

```

cur_succs = g.sample_successor(cur_nodes, max_degree=10, edge_type = "u2u" )

```

### 3. 获得节点特征get_node_feat

参数:

`nodes`: 查询节点

返回：

list of string： 每一行是对应节点的特征，字符串返回，如果没有特征则返回空。

```
feature = g.get_node_feat(nodes)

```

### 4. 高级负采样 sample_nodes

参数:

`nodes`: 需要给定采样节点

`neg_num`: 给每个节点采样的负样本个数

`neg_sample_type`: 如果 `neg_sample_type == 'm2v_plus'`，那么返回的负样本的节点类型会与输入nodes相同，其他情况下，全局随机负采样。

返回: np.array shape [len(nodes), neg_size]


```

# m2v_plus负采样
negs = g.sample_nodes(nodes, 10, "m2v_plus")

# 全局负采样
negs = g.sample_nodes(nodes, 10, "average")
 
```

### 5. 随机游走 metapath_randomwalk

注意：尽量配合node_batch_iter使用来遍历游走。

参数:

`graph`: 图客户端

`start_nodes`: 初始节点

`metapath`: meta-path 路径，例如 "u2i-i2u" 或者同够图下"u2u"

`walk_length`: 路径长度

返回:

一串路径

```
from pgl.distributed.dist_sample import metapath_randomwalk

for batch_node in g.node_batch_iter(batch_size,
                            node_type="u",
                            shuffle=True,
                            rank=rank,
                            nrank=nrank):

    # 同构图游走
    walks = metapath_randomwalk(graph=g,
                                start_nodes=batch_nodes,
                                metapath="u2u",
                                walk_length=24)
 

```
