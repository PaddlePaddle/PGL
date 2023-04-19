# PGLBox: Distributed Hierarchical GPU Engine for Efficiently Training Super-large Scale Graph Neural Network ([English document](./README_EN.md))

**PGLBox**是基于**GPU**的超大规模图模型训练引擎，通过异构层次化存储技术，突破了显存瓶颈，单机即可支持百亿节点、数百亿边的图采样和训练。用户只需要对配置文件进行简单的配置，便可以利用单机多卡训练大规模的图表示学习，并可快速搭建基于GNN的推荐系统、用户画像、图检索系统。

<h4 align="center">
  <a href=#快速开始> 快速开始 </a> |
  <a href=#数据格式> 数据格式 </a> |
  <a href=#特性> 特性 </a> |
  安装 |
  模型部署
</h4>

## 更新日志

- v2.2: 新增功能，包括增加带权采样功能、返回边权重功能、新增TransformerConv模型等。(2023.04.18)
- v2.1: 更新部分代码，包括增加[指定训练或infer节点](./wiki/train_infer_from_file_ch.md)、增加[例行训练功能](./wiki/online_train_ch.md)。(2023.03.08)
- v2.0: PGLBox V2版本，支持特征和Embedding多级存储，支持更大图规模。(2022.12.29)
- v1.0: 新增PGLBox能力，V1版本。(2022.12.14)


## 快速开始

为了可以快速使用PGLBox的能力，我们提供了一些相应的镜像环境，只需要拉取相关硬件的镜像，下载相应的数据，修改配置文件，就可以一键运行。目前PGLBox只支持在**v100**和**a100**这两款GPU硬件下运行。

```
docker pull registry.baidubce.com/paddlepaddle/pgl:pglbox-2.1-cuda11.0-cudnn8
```

拉取好docker之后，我们先下载PGLBox的代码，并进入PGLBox目录。

```
git clone https://github.com/PaddlePaddle/PGL
cd PGL/apps/PGLBox
```

进入目录后，只需放置一些我们训练图表示模型必要的图数据，例如节点编号文件以及边文件。数据格式后面会有详细介绍，这里提供一份大规模学术引用网络的数据MAG240M，以供快速开始。解压数据文件到当前目录。
```
wget https://baidu-pgl.gz.bcebos.com/pglbox/data/MAG240M/preprocessed_MAG240M.tar.gz
tar -zxf preprocessed_MAG240M.tar.gz
```
按照图的结构信息，以及需要的模型配置我们的配置文件，可以直接使用我们提供的[这份配置](./demo_configs/mag240m_metapath2vec.yaml)。具体配置含义我们会在后面进行解释。

在PGLBox主目录下通过`nvidia-docker run`命令运行模型
```
nvidia-docker run -it --rm \
    --name pglbox_docker \
    --network host \
    --ipc=host \
    -v ${PWD}:/pglbox \
    -w /pglbox \
    registry.baidubce.com/paddlepaddle/pgl:pglbox-2.1-cuda11.0-cudnn8 \
    /bin/bash -c "/pglbox/scripts/train.sh ./demo_configs/mag240m_metapath2vec.yaml"
```
训练完成后，我们可以在主目录下找到`mag240m_output`文件夹，该文件夹下包含了`model`和`embedding`两个文件夹，分别表示保存的模型以及infer产出的节点embedding。

## 数据格式

图数据格式与数据预处理的详细介绍请参考[这里](./wiki/data_format_ch.md)。


## 特性

#### <a href=#纯GPU框架的加速体验> 🚀 纯GPU框架的加速体验 </a>

#### <a href=#一键式配置化的复杂GNN模型支持>  📦 一键式配置化的复杂GNN模型支持 </a>

#### <a href=#提供丰富场景化解决方案> 📖 提供丰富场景化解决方案</a>


### 纯GPU框架的加速体验

在2021年底我们开源了[Graph4Rec](https://github.com/PaddlePaddle/PGL/tree/main/apps/Graph4Rec)工具库，主要用于大规模推荐场景下的图节点表示学习，该工具库主要用于多CPU场景下的大规模训练，并没有利用上GPU的快速计算能力。因此，今年我们开源了PGLBox纯GPU训练框架，将Graph4Rec全流程从CPU迁移到了GPU，大大提升了模型整体训练速度。（速度数据TBD）

### 一键式配置化的复杂GNN模型支持

在工业级图表示学习算法中，除了对图的规模要求高之外，还有复杂特征融合、游走策略、图聚合方式、算法组合多样化和例行训练等需求。我们延续Graph4Rec的设计策略，将这些现实问题，抽象成几个配置模块，即可完成复杂的GNN支持，适配**异构图神经网络**，**元路径随机游走**，**大规模稀疏特征**等复杂场景。我们也在`./user_configs`目录下提供了不同设置下的模型配置文件，供用户做选择。

<h2 align="center">
<img src="./../Graph4Rec/img/architecture.png" alt="graph4rec" width="800">
</h2>

总体来讲，完成自定义的配置，需要完成**图数据准备**，**图游走配置**，**GNN配置**，**训练参数配置**等。不同配置下，由于样本量和模型的计算复杂度不一样，耗时和效果差异也比较大，我们提供了一份在标准数据上，各个不同配置的耗时展示（TBD），以供参考。

<details><summary>图数据的准备</summary>

图数据的准备请参考[这里](./wiki/data_format_ch.md)。

默认情况下，PGLBox会训练图数据中所有节点并且预测出所有节点的embedding。如果用户只想训练部分节点，或者只预测部分节点，PGLBox提供了相应的功能支持，具体可以参考[这里](./wiki/train_infer_from_file_ch.md)

<br/>
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
`./user_configs`文件夹下查看具体yaml文件，里面对各个配置参数有更详细的解释。

### 提供丰富场景化解决方案 

下面我们给出若干使用**PGLBox**来完成的场景化案例，用户跟随场景教程，替换数据以及配置，即可完成相应的模型训练和部署。

- [在无属性图上的应用](./wiki/application_on_no_slot_features_ch.md)

- [在有属性图上的应用](./wiki/application_on_slot_features_ch.md)

- [在有边权重图上的应用](./wiki/application_on_edge_weight_ch.md)

- [在多种边类型图上的应用](./wiki/application_on_multi_edge_types_ch.md)
