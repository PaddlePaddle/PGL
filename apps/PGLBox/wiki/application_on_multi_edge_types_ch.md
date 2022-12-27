# PGLBox在多种边类型图上的应用

在电商异构图上，节点类型可能就只有user和item，但是边关系可以有多种，例如购买，点击，浏览，收藏等不同类型的关系。PGLBox支持在异构关系图上的游走与训练。

[UserBehavior](https://tianchi.aliyun.com/dataset/649)是阿里巴巴提供的一个淘宝用户行为数据集，用于隐式反馈推荐问题的研究。
本文使用UserBehavior数据集来演示如何在PGLBox上实现多种边关系的游走和训练。


## 数据下载

可以通过下面的链接下载UserBehavior数据集。

```

wget https://baidu-pgl.gz.bcebos.com/pglbox/data/UB/raw_UB.tar.gz

tar -zxf raw_UB.tar.gz

```

## 数据处理

假设我们把图数据解压到PGLBox目录下：

```
/xxx/PGLBox/raw_UB
    |
    |--node_types/
    |--u2buy2i/
    |--u2cart2i/
    |--u2click2i/
    |--u2fav2i/
```

这份数据集只有user（u）和item（i）两种节点类型，而有buy（购买），cart（购物车），click（点击）和fav（收藏）这四种类型的边关系。也就是在边文件中，每一行的左边都是u，右边都是i。为了区分不同边类型，我们这里使用“2”来分割三元组，u2buy2i表示（u，buy，i），以此类推。这样就可以把不同边关系区分开了。


接下来，使用我们提供的sharding_tool工具，将节点类型文件和边类型文件进行分片。详细使用方法请参考[这里](https://github.com/PaddlePaddle/PGL/tree/main/apps/PGLBox/sharding_tool/)。


### 配置文件设置

对图数据进行分片处理后，我们假设处理好的图数据保存在这个目录下`/your/path/to/preprocessed_UB`，即：

```bash
/xxx/PGLBox/preprocessed_UB
    |
    |--node_types/
    |--u2buy2i/
    |--u2cart2i/
    |--u2click2i/
    |--u2fav2i/
```

那么接下来我们只需要在配置文件中设定以下几个参数，就可以换成这份处理好的图数据来训练了。下面我们演示修改`./user_configs/metapath.yaml`配置文件的以下几个参数来实现图数据的替换。

```bash
graph_data_local_path: "/pglbox/preprocessed_UB"

etype2files: "u2buy2i:u2buy2i,u2cart2i:u2cart2i,u2click2i:u2click2i,u2fav2i:u2fav2i,"

ntype2files: "u:node_types,i:node_types"

hadoop_shard: True

num_part: 1000

symmetry: True
```

**注意**: 一般我们是用docker容器来训练模型，那么要注意`graph_data_local_path`在docker容器里面的路径映射关系。不然会因为数据路径错误导致读取不到图数据。在上面的配置中，之所以使用`graph_data_local_path: "/pglbox/preprocessed_UB"`，是因为我们下面使用docker训练的时候，把当前目录映射到了`/pglbox`目录了。


### 图游走配置

数据配置完成后，就可以对图游走进行配置了。具体配置如下：

```
meta_path: "u2buy2i-i2buy2u;u2click2i-i2click2u;u2cart2i-i2cart2u;i2cart2u-u2cart2i;i2buy2u-u2buy2i;i2click2u-u2click2i;u2fav2i-i2fav2u;i2fav2u-u2fav2i"


```

可以看到，这里除了有u2buy2i 类型外，还出现了i2buy2u这种类型。这是因为训练的时候我们都是使用无向边（双向边），这样游走的时候才可以往复循环游走。用户的数据只需要提供u2buy2i类型，反向类型的数据构造在使用[sharding_tool](https://github.com/PaddlePaddle/PGL/tree/main/apps/PGLBox/sharding_tool/)工具分片处理的时候会自动构造。


### 训练

对于`./user_configs/`目录下的所有配置文件，只需要修改完上述的配置后，即可运行了。下面以`user_configs/metapath.yaml`为例进行演示：

```

nvidia-docker run -it --rm \
    --name pglbox_docker \
    --network host \
    --ipc=host \
    -v ${PWD}:/pglbox \
    -w /pglbox \
    registry.baidubce.com/paddlepaddle/pgl:pglbox-2.0rc-cuda11.0-cudnn8 \
    /bin/bash -c "/pglbox/train.sh /pglbox/user_configs/metapath.yaml"
    
```
