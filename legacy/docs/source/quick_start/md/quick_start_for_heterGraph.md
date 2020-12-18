## Introduction

In real world, there exists many graphs contain multiple types of nodes and edges, which we call them Heterogeneous Graphs. Obviously, heterogenous graphs are more complex than homogeneous graphs. 

To deal with such heterogeneous graphs, PGL develops a graph framework to support graph neural network computations and meta-path-based sampling on heterogenous graph.

The goal of this tutorial:
* example of heterogenous graph data;
* Understand how PGL supports computations in heterogenous graph;
* Using PGL to implement a simple heterogenous graph neural network model to classfiy a particular type of node in a heterogenous graph network.

## Example of heterogenous graph

There are a lot of graph data that consists of edges and nodes of multiple types. For example, **e-commerce network** is very common heterogenous graph in real world. It contains at least two types of nodes (user and item) and two types of edges (buy and click). 

The following figure depicts several users click or buy some items. This graph has two types of nodes corresponding to "user" and "item". It also contain two types of edge "buy" and "click".
![A simple heterogenous e-commerce graph](images/heter_graph_introduction.png)

## Creating a heterogenous graph with PGL 

In heterogenous graph, there exists multiple edges, so we should distinguish them. In PGL, the edges are built in below format:
```python
edges = {
    'click': [(0, 4), (0, 7), (1, 6), (2, 5), (3, 6)],
    'buy': [(0, 5), (1, 4), (1, 6), (2, 7), (3, 5)],
        }
```

In heterogenous graph, nodes are also of different types. Therefore, you need to mark the type of each node, the format of the node type is as follows:

```python
node_types = [(0, 'user'), (1, 'user'), (2, 'user'), (3, 'user'), (4, 'item'), 
             (5, 'item'),(6, 'item'), (7, 'item')]
```

Because of the different types of edges, edge features also need to be separated by different types.

```python
import numpy as np

num_nodes = len(node_types)

node_features = {'features': np.random.randn(num_nodes, 8).astype("float32")}

edge_num_list = []
for edge_type in edges:
    edge_num_list.append(len(edges[edge_type]))

edge_features = {
    'click': {'h': np.random.randn(edge_num_list[0], 4)},
    'buy': {'h':np.random.randn(edge_num_list[1], 4)},
}
```

Now, we can build a heterogenous graph by using PGL.

```python
import paddle.fluid as fluid
import paddle.fluid.layers as fl
import pgl
from pgl import heter_graph
from pgl import heter_graph_wrapper

g = heter_graph.HeterGraph(num_nodes=num_nodes,
                            edges=edges,
                            node_types=node_types,
                            node_feat=node_features,
                            edge_feat=edge_features)
```



In PGL, we need to use graph_wrapper as a container for graph data, so here we need to create a graph_wrapper for each type of edge graph.

```python
place = fluid.CPUPlace()

# create a GraphWrapper as a container for graph data
gw = heter_graph_wrapper.HeterGraphWrapper(name='heter_graph', 
                                    edge_types = g.edge_types_info(),
                                    node_feat=g.node_feat_info(),
                                    edge_feat=g.edge_feat_info())
```



## MessagePassing

After building the heterogeneous graph, we can easily carry out the message passing mode. In this case, we have two different types of edges, so we can write a function in such way:

```python
def message_passing(gw, edge_types, features, name=''):
    def __message(src_feat, dst_feat, edge_feat): 
        return src_feat['h']
    def __reduce(feat):
        return fluid.layers.sequence_pool(feat, pool_type='sum')
    
    assert len(edge_types) == len(features)
    output = []
    for i in range(len(edge_types)):
        msg = gw[edge_types[i]].send(__message, nfeat_list=[('h', features[i])])
        out = gw[edge_types[i]].recv(msg, __reduce)  
        output.append(out)
    # list of matrix
    return output
```

```python
edge_types = ['click', 'buy']
features = []
for edge_type in edge_types:
    features.append(gw[edge_type].node_feat['features'])
output = message_passing(gw, edge_types, features)

output = fl.concat(input=output, axis=1)

output = fluid.layers.fc(output, size=4, bias_attr=False, act='relu', name='fc1')
logits = fluid.layers.fc(output, size=1, bias_attr=False, act=None, name='fc2')
```



## data preprocessing 

In this case, we implement a simple node classifier, we can use 0,1 to represent two classes.

```python
y = [0,1,0,1,0,1,1,0]  
label = np.array(y, dtype="float32").reshape(-1,1)
```



## Setting up the training program
The training process of the heterogeneous graph node classification model is the same as the training of other paddlepaddle-based models.
* First we build the loss function;
* Second, creating a optimizer;
* Finally, creating a executor and execute the training program.

```python
node_label = fluid.layers.data("node_label", shape=[None, 1], dtype="float32", append_batch_size=False)


loss = fluid.layers.sigmoid_cross_entropy_with_logits(x=logits, label=node_label)

loss = fluid.layers.mean(loss)


adam = fluid.optimizer.Adam(learning_rate=0.01)
adam.minimize(loss)


exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
feed_dict = gw.to_feed(g) 

for epoch in range(30):
    feed_dict['node_label'] = label
    
    train_loss = exe.run(fluid.default_main_program(), feed=feed_dict, fetch_list=[loss], return_numpy=True)
    print('Epoch %d | Loss: %f'%(epoch, train_loss[0]))
```
