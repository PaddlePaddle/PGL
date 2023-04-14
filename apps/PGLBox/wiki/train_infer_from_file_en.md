# Train or infer the specified node file

In order to meet the user's needs for only training some nodes, or only infering some nodes, PGLBox provides corresponding functional supports.

## Training specified node file directory

When configuring the yaml file, the configuration item `train_start_nodes` is used to specify the starting node directory for training. If it is left blank, the node that originally loaded the graph data will be used for training by default. If needed, a simple example is as follows:

``` shell
graph_data_local_path: /your/graph_data_path/
train_start_nodes: "train_nodes"
```

The above configuration items mean that the directory structure is as follows:
``` shell
/your/graph_data_path/
|--user2item
|--node_types
|--train_nodes
```

Among them, the file format in the `train_nodes` folder needs to be consistent with `node_types`:w
. If the `hadoop_shard` config is True, it also needs to be divided into 1000 parts.

## Inference specified node file directory

When configuring the yaml file, the configuration item `infer_nodes` is used to specify the starting node directory for inference. If it is left blank, the node that originally loaded the graph data will be used for infering by default. If needed, a simple example is as follows:

``` shell
graph_data_local_path: /your/graph_data_path/
infer_nodes: "infer_nodes"
```

The above configuration items mean that the directory structure is as follows:
``` shell
/your/graph_data_path/
|--user2item
|--node_types
|--infer_nodes
```
