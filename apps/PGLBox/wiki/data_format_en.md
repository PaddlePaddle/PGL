# Graph Data Format

This document mainly introduces the graph data format required for graph representation learning and training using PGLBox.

Suppose we have three types of nodes: author, paper and inst. There are three types of edges `author2paper` (author - write - paper), `author2inst`(author - belong to - inst) and `paper2paper`(paper - cite - paper).

Then we need to prepare node files and edge files.

## Node Files

The format of node files is:

```
node_type \t node_id
```
Among them, node_type is the type of the node, such as paper, author or inst. And node_id is a number of **uint64**, **but note that it cannot be 0**.

```bash

# an example

inst	523515
inst	614613
inst	611434
paper	2342353241
paper	2451413511
author	9905123492
author	9845194235

```

## Node file with slot features

If the node has discrete slot features, the format of the node type file is:
```bash

node_type \t node_id \t slot1:slot1_value \t slot2:slot2_value \t slot3:slot3_value

```

Then in the configuration file in the `./user_configs/` directory, modify the slots parameter to:
```bash
slots: ["slot1", "slot2", "slot3"]

```
Note that these slots are numbers. Different slots represent different feature types. For example, "11" means gender, "21" means age, etc.

For example, the author type node has two slot features of gender and age, represented by "11" and "21" respectively. Paper type nodes have the slot feature of publication year, which is represented by "101". Then the node type file can be expressed as:

```bash
# an example

inst	523515
inst	614613
inst	611434
paper	2342353241    101:2020
paper	2451413511    101:2019
author	9905123492    11:1     21:30
author	9845194235    11:0     21:24

```

In the configuration file, the slots parameter is set to:
```bash
slots: ["11", "21", "101"]
```

## Edge file

Each edge type requires a separate directory to hold edge relationships. For example, there are three edge types here, so three directories are needed here.

The format of the edge file is:
```
src_node_id \t dst_node_id

```
Among them, these `src_node_id` and `dst_node_id` are **uint64** numbers.

```

# an example: author2paper

9905123492	2451413511
9845194235	2342353241
9845194235	2451413511 

```

## Data Directory Structure

According to the above data format for data preparation, we can get the following four data subdirectories:

```bash
/your/path/to/graph_data
    |
    |--node_types/
    |--author2paper/
    |--paper2paper/
    |--author2inst/
```
Among them, the `node_types` subdirectory saves the node type and the corresponding node slot characteristics (if any). The `author2paper` subdirectory saves (author - write - paper) type edges, the `paper2paper` subdirectory saves (paper - cite - paper) type edges, and the `author2inst` subdirectory saves (author - belong to - inst) type of edge.


## Graph Data Sharding

In order to speed up the loading speed of graph data, we need to slice the graph data. After sharding, the graph engine can load in parallel with multiple threads, which greatly improves the speed of loading graph data.

Therefore, you can use the shardind tool we provide to shard the node files and edge files. Please refer to [here](https://github.com/PaddlePaddle/PGL/tree/main/apps/PGLBox/sharding_tool/) for detailed usage.


## Configuration Settings

After sharding the graph data, we assume that the processed graph data is saved in this directory `/your/path/to/preprocessed_graph_data`, namely:
```bash
/your/path/to/preprocessed_graph_data
    |
    |--node_types/
    |--author2paper/
    |--paper2paper/
    |--author2inst/
```

Then we only need to set the following parameters in the configuration file, and then we can replace it with this processed graph data for training:

```bash
graph_data_local_path: "/your/path/to/preprocessed_graph_data"

etype2files: "author2paper:author2paper,paper2paper:paper2paper,author2inst:author2inst"

ntype2files: "author:node_types,paper:node_types,inst:node_types"

hadoop_shard: True

num_part: 1000

symmetry: True
```

**Note**: Generally, we use the docker container to train the model, so pay attention to the path mapping relationship of `graph_data_local_path` in the docker container. Otherwise, the graph data will not be read due to data path errors.
