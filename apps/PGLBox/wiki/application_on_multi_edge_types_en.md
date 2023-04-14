# Application of PGLBox on Multiple Edge Type Graphs

On the e-commerce heterogeneous graph, the node types may only be user and item, but there can be many kinds of edge relationships, such as purchase, click, browse, favorite and other types of relationships. PGLBox supports walking and training on heterogeneous graphs.

[UserBehavior](https://tianchi.aliyun.com/dataset/649) is a Taobao user behavior dataset provided by Alibaba, which is used for the research of implicit feedback recommendation problems. This article uses the UserBehavior dataset to demonstrate how to walk and train multiple edge relationships on PGLBox.

## Data Download

The UserBehavior dataset can be downloaded via the link below.

```
wget https://baidu-pgl.gz.bcebos.com/pglbox/data/UB/raw_UB.tar.gz

tar -zxf raw_UB.tar.gz

```

## Data Process

Suppose we decompress the graph data into the PGLBox directory:

```
/xxx/PGLBox/raw_UB
    |
    |--node_types/
    |--u2buy2i/
    |--u2cart2i/
    |--u2click2i/
    |--u2fav2i/
```

This data set has only two node types, user (u) and item (i), and four types of edge relationships: buy (purchase), cart (shopping cart), click (click) and fav (collection). That is, in the edge file, each line has u on the left and i on the right. In order to distinguish different edge types, we use "2" here to split triples, u2buy2i means (u, buy, i), and so on. In this way, different edge types can be distinguished.

Next, use the sharding tool we provide to shard the node type files and edge type files. Please refer to [here](https://github.com/PaddlePaddle/PGL/tree/main/apps/PGLBox/sharding_tool/) for detailed usage.

## Configuration Settings

After sharding the graph data, we assume that the processed graph data is saved in this directory `/your/path/to/preprocessed_UB`, namely:

```bash
/xxx/PGLBox/preprocessed_UB
    |
    |--node_types/
    |--u2buy2i/
    |--u2cart2i/
    |--u2click2i/
    |--u2fav2i/
```

Then we only need to set the following parameters in the configuration file, and then we can replace it with this processed graph data for training. Below we demonstrate how to modify the following parameters of the `./user_configs/metapath.yaml` configuration file to replace graph data.

```bash
graph_data_local_path: "/pglbox/preprocessed_UB"

etype2files: "u2buy2i:u2buy2i,u2cart2i:u2cart2i,u2click2i:u2click2i,u2fav2i:u2fav2i,"

ntype2files: "u:node_types,i:node_types"

hadoop_shard: True

num_part: 1000

symmetry: True
```

**Note**: Generally, we use the docker container to train the model, so pay attention to the path mapping relationship of `graph_data_local_path` in the docker container. Otherwise, the graph data will not be read due to data path errors. In the above configuration, the reason why `graph_data_local_path: "/pglbox/preprocessed_UB"` is used is that when we use docker training below, the current directory is mapped to the `/pglbox` directory.


## Graph traversal configuration

After the data configuration is complete, the graph traversal parameters can be configured. The specific configuration is as follows:

```
meta_path: "u2buy2i-i2buy2u;u2click2i-i2click2u;u2cart2i-i2cart2u;i2cart2u-u2cart2i;i2buy2u-u2buy2i;i2click2u-u2click2i;u2fav2i-i2fav2u;i2fav2u-u2fav2i"

```

It can be seen that in addition to the u2buy2i type, there is also the i2buy2u type. This is because we should use undirected edges (bidirectional edges) during training, so that we can walk back and forth when walking. The user's data only needs to provide the u2buy2i type, and the reverse type of data structure is processed when using [sharding_tool](https://github.com/PaddlePaddle/PGL/tree/main/apps/PGLBox/sharding_tool/), which will be constructed automatically.


## Training

For all configuration files in the `./user_configs/` directory, you only need to modify the above configurations before running. Let's take `user_configs/metapath.yaml` as an example.

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
