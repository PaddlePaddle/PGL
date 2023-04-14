# 例行训练

PGLBox支持例行化训练，需要依赖用户使用hadoop集群或者本机进行图数据的存储以及模型的存储等。例行化训练的意思本质上是每隔一个时间段后训练新的数据，这个过程中保持任务常驻，而无需重启整个任务。

通过在配置文件中设定好`train_mode`、`start_time` 和 `time_delta`这三个参数，就可以从指定的start_time对应的图数据开始训练，按照time_delta的时间间隔进行例行化训练而无需退出任务。（注意如果为hadoop集群，需要配置多个参数，包括`hadoop_home`、`graph_data_fs_name`、`graph_data_fs_ugi`、`output_fs_name`和`output_fs_ugi`，具体含义可以看配置yaml文件。）

如果没有新数据，该任务会在线等待下一个时间的图数据产生。

## 图数据准备

我们的图数据需要按照 **/your/graph_data_path/日期/小时** 的格式存放。下面的例子展示了从 start_time=20230219/02 开始，time_delta=3 小时的数据产生过程。

``` shell
# 可以是hadoop路径，也可以是本机相对于docker容器下的路径。
/your/graph_data_path/20230219/02
/your/graph_data_path/20230219/05
/your/graph_data_path/20230219/08
/your/graph_data_path/20230219/11
/your/graph_data_path/20230219/14
```

Notes: 需要注意的是，当某个delta数据上传完毕后，需要在对应的文件夹下生成 **to.hadoop.done** 文件，以说明对应delta数据已上传完成，才可以进行后续的加载与训练。

有了上面存放图数据的格式之后，我们在配置文件中只需要修改如下参数：
``` shell
# 此处，我们通过判断路径的开头是否包含`afs`或者`hdfs`来区分为hadoop路径还是本机路径。
graph_data_hdfs_path: /your/graph_data_path/

start_time: 20230219/02
time_delta: 3
```

经过上述配置之后，在第一次冷启动在线训练时，就会从 start_time=20230219/02 的目录下加载图数据进行训练。随后，在训练完一个新的图数据目录之前，代码会自动往config.yaml文件中写入如 newest_time: 20230219/05，用于指示下一个训练时需要加载的图数据目录。

## 图模型的保存

每次图数据训练完成后，会把训练好的模型按照 **/your/graph_model_path/日期/小时** 的格式进行保存，保存的目录格式与图数据的格式一致。 如下所示：
``` shell
# 可以是hadoop路径，也可以是本机相对于docker容器下的路径。
/your/graph_model_path/20230219/02
/your/graph_model_path/20230219/05
/your/graph_model_path/20230219/08
/your/graph_model_path/20230219/11
```

每个目录下保存了训练好的模型以及产出的节点Embedding，对应文件夹名为model和embedding。每次训练完一个time_delta的图数据后，我们会在config.yaml文件中对 warm_start_from 配置项进行修改，用于指示下一个训练时需要热启的模型目录。
