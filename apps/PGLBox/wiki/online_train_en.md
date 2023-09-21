# Routine Training

PGLBox supports routine training, which needs to rely on users to use hadoop clusters or local machines to store graph data and models. Routine training means to train new data every other time period, and keep the task resident during this process without restarting the entire task.

By setting the three parameters `train_mode`, `start_time` and `time_delta` in the configuration file, you can start training from the graph data corresponding to the specified start_time, and perform routine training according to the time interval of time_delta without exiting Task. (Note that if it is a hadoop cluster, multiple parameters need to be configured, including `hadoop_home`, `graph_data_fs_name`, `graph_data_fs_ugi`, `output_fs_name` and `output_fs_ugi`, the specific meaning can be found in the configuration yaml file.)

If there is no new data, the task will wait online to generate graph data at the next time.

## Graph Data Preparation

Our graph data needs to be stored in the format of **/your/graph_data_path/date/hour**. The following example shows the data generation process starting from start_time=20230219/02 and time_delta=3 hours.

``` shell
# It can be the hadoop path, or the local path relative to the docker container.
/your/graph_data_path/20230219/02
/your/graph_data_path/20230219/05
/your/graph_data_path/20230219/08
/your/graph_data_path/20230219/11
/your/graph_data_path/20230219/14
```

Notes: It should be noted that after a certain delta data is uploaded, a **to.hadoop.done** file needs to be generated in the corresponding folder to indicate that the corresponding delta data has been uploaded, then we can use this data to train.

With the above format for storing graph data, we only need to modify the following parameters in the configuration file:

``` shell
# Here, we distinguish whether it is a hadoop path or a local path by judging whether the beginning of the path contains `afs` or `hdfs`.
graph_data_hdfs_path: /your/graph_data_path/

start_time: 20230219/02
time_delta: 3
```

After the above configuration, when the online training is cold started for the first time, the graph data will be loaded from the directory `start_time=20230219/02` for training. Then, before training a new graph data directory, the code will automatically write `newest_time: 20230219/05` to the config.yaml file to indicate the graph data directory that needs to be loaded during the next training.


## Model Save

After each graph data training is completed, the trained model will be saved in the format of **/your/graph_model_path/date/hour**, and the format of the saved directory is consistent with the format of the graph data. As follows:

``` shell
# It can be the hadoop path, or the local path relative to the docker container.
/your/graph_model_path/20230219/02
/your/graph_model_path/20230219/05
/your/graph_model_path/20230219/08
/your/graph_model_path/20230219/11
```

Each directory saves the trained model parameters and the output node embeddings, where the corresponding folder names are `model` and `embedding`. After training a time_delta graph data, we will modify the `warm_start_from` configuration item in the config.yaml file to indicate the model directory that needs to be warm-started for the next training.

