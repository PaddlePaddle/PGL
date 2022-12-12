# 多进程切分图数据

## 简介

PGLBox在加载图数据的时候，如果没有提前对数据进行分片切分，那么它只能单线程加载。
在数据量不大（千万节点，一亿边）的情况下还能忍受。
但是一旦数据量更大，就会使得图数据加载时间总体变得很长。

为了减少图数据加载时间，我们可以先对图数据进行分片，
然后在加载数据的时候，PGLBox就可以使用多线程读取数据，从而提高加载图数据的速度。

这个sharding工具就是提前对图数据进行分片，以提高PGLBox加载图数据的速度。

## 使用方法

### 配置说明

在运行之前，需要修改`./run_sharding.sh`文件中的一些参数，具体如下：

* part_num: 需要分片的数量，目前只能是1000

* base_input_dir: 原始图数据所在目录

* base_output_dir: 分片后的数据保存目录


### 举例说明

举个例子，我们假设原始图数据在`/your/path/to/graph_data`目录。此目录下有四个子目录，分别是：

```
/your/path/to/graph_data
    |
    |--node_types/
    |--author2paper/
    |--paper2paper/
    |--author2inst/
```

最后希望分片完的数据输出到`/your/path/to/preprocessed_graph_data`，即：

```
/your/path/to/preprocessed_graph_data
    |
    |--node_types/
    |--author2paper/
    |--paper2paper/
    |--author2inst/
```

那么, 我们在`run_sharding.sh`脚本中，则需要设置如下命令：

```

part_num=1000
base_input_dir="/your/path/to/graph_data"
base_output_dir="/your/path/to/preprocessed_graph_data"

# sharding node_types
python graph_sharding.py --input_dir ${base_input_dir}/node_types \
                         --output_dir ${base_output_dir}/node_types \
                         --part_num ${part_num} \
                         --node_type_shard

for etype in author2paper paper2paper author2inst; do
    python graph_sharding.py --input_dir ${base_input_dir}/${etype} \
                             --output_dir ${base_output_dir}/${etype} \
                             --part_num ${part_num} \
                             --symmetry
done

```

如果有更多的边类型，则以此类推在for循环中增加即可。

### 运行

完成上述配置后，运行下面的命令即可对图数据进行分片。

```
sh run_sharding.sh
```
