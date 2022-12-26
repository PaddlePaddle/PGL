
## PGLBox: Distributed Hierarchical GPU Engine for Efficiently Training Super-large Scale Graph Neural Network


**PGLBox**是基于**GPU**的超大规模图模型训练引擎，通过异构层次化存储技术，突破了显存瓶颈，单机即可支持百亿节点、数百亿边的采样和训练。通过PGLBox你可以简单地通过配置，利用单机多卡训练大规模的图表示学习，以此来搭建快速基于GNN的推荐系统、用户画像、图检索系统。

<h4 align="center">
  <a href=#快速开始> 快速开始 </a> |
  <a href=#特性> 特性 </a> |
  安装 |
  模型部署
</h4>

## 快速开始

为了我们可以快速使用PGLBox的能力，我们提供了一些相应的镜像环境，只需要拉取相关硬件的镜像，然后完成数据的配置，就可以一键跑起图模型。
```
docker pull registry.baidubce.com/paddlepaddle/pgl:pglbox-2.0rc-cuda11.0-cudnn8
```
拉取好docker之后，我们先下载PGLBox的代码，并进入PGLBox目录
```
git clone https://github.com/PaddlePaddle/PGL
cd PGL/apps/PGLBox
```
放置一些我们训练图表示模型必要的数据，例如节点编号文件以及边文件，详情可见后续介绍。这里提供一份大规模学术引用网络的数据MAG240M，以供快速测试。解压数据文件到当前目录。
```
wget https://baidu-pgl.gz.bcebos.com/pglbox/data/MAG240M/preprocessed_MAG240M.tar.gz
tar -zxf preprocessed_MAG240M.tar.gz
```
按照图的结构信息，以及需要的模型配置我们的配置文件，可以直接使用我们提供的这份配置。具体配置含义我们会在后面进行解释。
```
wget https://baidu-pgl.gz.bcebos.com/pglbox/data/MAG240M/mag240m_metapath2vec.yaml
```
在PGLBox主目录下通过`nvidia-docker run`命令运行模型
```
nvidia-docker run -it --rm \
    --name pglbox_docker \
    --network host \
    --ipc=host \
    -v ${PWD}:/pglbox \
    -w /pglbox \
    registry.baidubce.com/paddlepaddle/pgl:pglbox-2.0rc-cuda11.0-cudnn8 \
    /bin/bash -c "/pglbox/train.sh ./mag240m_metapath2vec.yaml"
```
训练完成后，我们可以在主目录下找到`mag240m_output`文件夹，该文件夹下包含了`model`和`embedding`两个文件夹，分别表示保存的模型以及infer产出的节点embedding。

## 特性

#### <a href=#纯GPU框架的加速体验> 🚀 纯GPU框架的加速体验 </a>

#### <a href=#一键式配置化的复杂GNN模型支持>  📦 一键式配置化的复杂GNN模型支持 </a>

#### <a href=#提供丰富场景化解决方案> 📖 提供丰富场景化解决方案</a>

### 纯GPU框架的加速体验

在2021年底我们开源了[Graph4Rec](https://github.com/PaddlePaddle/PGL/tree/main/apps/Graph4Rec)工具库，主要用于大规模推荐场景下的图节点表示学习，该工具库主要用于多CPU场景下的大规模训练，并没有利用上GPU的快速计算能力。因此，今年我们开源了PGLBox纯GPU训练框架，将Graph4Rec全流程从CPU迁移到了GPU，大大提升了模型整体训练速度。（速度数据TBD）

### 一键式配置化的复杂GNN模型支持
在工业级图表示学习算法中，除了对图的规模要求高之外，还有复杂特征融合、游走策略、图聚合方式、算法组合多样化和例行训练等需求。我们延续Graph4Rec的设计策略，将这些现实问题，抽象成几个配置模块，即可完成复杂的GNN支持，适配**异构图神经网络**，**元路径随机游走**，**大规模稀疏特征**等复杂场景。我们也在`user_configs`里面提供了不同设置下的模型配置，供用户做选择。

<h2 align="center">
<img src="./../Graph4Rec/img/architecture.png" alt="graph4rec" width="800">
</h2>

总体来讲，完成自定义的配置，需要完成**图数据准备**，**图游走配置**，**GNN配置**，**训练参数配置**等。不同配置下，由于样本量和模型的计算复杂度不一样，耗时和效果差异也比较大，我们提供了一份在标准数据上，各个不同配置的耗时展示（TBD），以供参考。

<details><summary>图数据的准备</summary>
<br/>
以MAG240M数据为例，其节点规模为2.44亿，边规模为17.28亿（不含对称边）。其中节点类型有paper、author、inst共计三种类型的节点。边则有paper2paper、author2inst、author2paper共计三种边。 那么我们需要有节点类型文件和边文件。

#### 节点类型文件准备

节点类型文件的格式为:
``` shell
node_type \t node_id
```
其中，`node_type`是节点的类型，例如为paper、author、inst。而`node_id`为**uint64**的数字，**但注意不能为0**。
``` shell
# 简单举例
paper	2342353241
paper	2451413511
author	190512349
author	9845194235
inst	523515
inst	6146134
inst	611434
```

如果节点带slot特征，则节点类型文件的格式为：
``` shell
node_type \t node_id \t slot1:slot1_value \t slot2:slot2_value \t slot3:slot3_value
```
然后在`./user_configs/`目录的配置文件中，修改`slots`参数为：
```
slots: ["slot1", "slot2", "slot3"]
```
注意，这些slot都是数字，不同的slot表示不同的特征类型。比如"11"表示性别，"21"表示年龄。

#### 边文件准备

边文件的格式为：
``` shell
src_node_id \t dst_node_id
```
其中，这些`src_node_id`和`dst_node_id`均为**uint64**的数字。
``` shell
# 简单举例
# paper2paper文件

2342353241	2451413511

# author2inst文件

190512349	523515
9845194235	6146134

# author2paper文件

190512349	2451413511
9845194235	2342353241
9845194235	2451413511 

```

**图数据分片**

为了能够加快图数据的加载速度，我们需要对图数据进行分片，分片之后图引擎就可以多线程并行加载，大大提高加载图数据的速度。因此，可以使用我们提供的[sharding_tool](https://github.com/PaddlePaddle/PGL/tree/main/apps/PGLBox/sharding_tool)工具，将节点类型文件和边类型文件进行分片。详细使用方法可以到对应链接的文档下查看。

注：我们所提供的`preprocessed_MAG240M`图数据已经是分片过的。

</details>

<details><summary>图游走配置</summary>
<br/>
图游走配置项主要用于控制图游走模型的具体参数。具体如下。

``` shell
# meta_path参数，配置图上的游走路径，这里我们以MAG240M图数据为例。
meta_path: "author2inst-inst2author;author2paper-paper2author;inst2author-author2paper-paper2author-author2inst;paper2paper-paper2author-author2paper"

# 表示游走路径的正样本窗口大小
win_size: 3

# 表示每对正样本对应的负样本数量
neg_num: 5

# meapath 游走路径的深度
walk_len: 24

# 每个起始节点重复walk_times次游走，这样可以尽可能把一个节点的所有邻居游走一遍，使得训练更加均匀。
walk_times: 10
```

</details>

<details><summary>GNN配置</summary>
<br/>
上述图游走配置主要是针对metapath2vec这类模型的配置项，在其基础之上，如果我们想要训练更为复杂的GNN图网络，则可以设置GNN网络的相关配置项进行模型调整。

``` shell
# GNN模型开关
sage_mode: True

# 不同GNN模型选择，包括LightGCN、GAT、GIN等，详细可看PGLBox的模型文件夹。
sage_layer_type: "LightGCN"

# 节点Embedding自身权重配比( sage_alpha )与GNN聚合后节点Embedding配比( 1 - sage_alpha ) 
sage_alpha: 0.9

# 训练时图模型采样节点邻居个数
samples: [5]

# infer时图模型采样节点邻居个数
infer_samples: [100]

# GNN模型激活层选择
sage_act: "relu"

```

</details>

<details><summary>模型训练参数配置</summary>
<br/>
除了上述一些配置外，这里还简单罗列一些相对比较重要的配置项。

``` shell
# 模型类型选择，目前默认不改动。后续我们会提供更多的选择，如ErnieSageModel等。
model_type: GNNModel

# embedding维度。
embed_size: 64

# 稀疏参数服务器的优化器，目前支持adagrad、shared_adam。
sparse_type: adagrad

# 稀疏参数服务器的学习率
sparse_lr: 0.05

# 损失函数，目前支持hinge、sigmoid、nce。
loss_type: nce

# 是否要进行训练，如果只想单独热启模型做预估(inference)，则可以关闭need_train。
need_train: True

# 是否需要进行inference. 如果只想单独训练模型，则可以关闭need_inference。
need_inference: True

# 训练轮数
epochs: 1

# 训练样本的batch_size
batch_node_size: 80000

# infer样本的batch_size
infer_batch_size: 80000

# 触发ssd cache的频率
save_cache_frequency: 4

# 在内存中缓存多少个pass数据集
mem_cache_passid_num: 4

# 训练模式，可填WHOLE_HBM/MEM_EMBEDDING/SSD_EMBEDDING，默认为MEM_EMBEDDING
train_storage_mode: MEM_EMBEDDING
```

</details>

除了以上这些配置参数以外，还有其他有关于数据配置、slot特征配置、模型保存配置等内容，更具体的信息可以到我们所提供的
`user_configs`文件夹下查看具体yaml文件，里面对各个配置参数有更详细的解释。

### 提供丰富场景化解决方案 

下面我们给出若干使用**PGLBox**来完成的场景化案例，用户跟随场景教程，替换数据以及配置，即可完成相应的模型训练和部署。

TBD
