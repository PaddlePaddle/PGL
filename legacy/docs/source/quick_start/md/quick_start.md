## Step 1: using PGL to create a graph 
Suppose we have a graph with 10 nodes and 14 edges as shown in the following figure:
![A simple graph](images/quick_start_graph.png)

Our purpose is to train a graph neural network to classify yellow and green nodes. So we can create this graph in such way:
```python
import pgl
from pgl import graph  # import pgl module
import numpy as np

def build_graph():
    # define the number of nodes; we can use number to represent every node
    num_node = 10
    # add edges, we represent all edges as a list of tuple (src, dst)
    edge_list = [(2, 0), (2, 1), (3, 1),(4, 0), (5, 0), 
             (6, 0), (6, 4), (6, 5), (7, 0), (7, 1),
             (7, 2), (7, 3), (8, 0), (9, 7)]

    # Each node can be represented by a d-dimensional feature vector, here for simple, the feature vectors are randomly generated.
    d = 16
    feature = np.random.randn(num_node, d).astype("float32")
    # each edge has it own weight
    edge_feature = np.random.randn(len(edge_list), 1).astype("float32")
    
    # create a graph
    g = graph.Graph(num_nodes = num_node,
                    edges = edge_list, 
                    node_feat = {'feature':feature}, 
                    edge_feat ={'edge_feature': edge_feature})

    return g

# create a graph object for saving graph data
g = build_graph()
```
After creating a graph in PGL, we can print out some information in the graph.

```python
print('There are %d nodes in the graph.'%g.num_nodes)
print('There are %d edges in the graph.'%g.num_edges)

# Out:
# There are 10 nodes in the graph.
# There are 14 edges in the graph. 
```

Currently our PGL is developed based on static computational mode of paddle (we’ll support dynamic computational model later). We need to build model upon a virtual data holder. GraphWrapper provide a virtual graph structure that users can build deep learning models based on this virtual graph. And then feed real graph data to run the models.
```python
import paddle.fluid as fluid

use_cuda = False  
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

# use GraphWrapper as a container for graph data to construct a graph neural network
gw = pgl.graph_wrapper.GraphWrapper(name='graph',
                        node_feat=g.node_feat_info(), 
                        edge_feat=g.edge_feat_info())
```

## Step 2: create a simple Graph Convolutional Network(GCN)

In this tutorial, we use a simple Graph Convolutional Network(GCN) developed by [Kipf and Welling](https://arxiv.org/abs/1609.02907) to perform node classification. Here we use the simplest GCN structure. If readers want to know more about GCN, you can refer to the original paper.

* In layer $l$，each node $u_i^l$ has a feature vector $h_i^l$;
* In every layer,  the idea of GCN is that the feature vector $h_i^{l+1}$ of each node $u_i^{l+1}$ in the next layer are obtained by weighting the feature vectors of all the neighboring nodes and then go through a non-linear transformation.  

In PGL, we can easily implement a GCN layer as follows:
```python
# define GCN layer function
def gcn_layer(gw, nfeat, efeat, hidden_size, name, activation):
    # gw is a GraphWrapper；feature is the feature vectors of nodes
    
    # define message function
    def send_func(src_feat, dst_feat, edge_feat): 
        # In this tutorial, we return the feature vector of the source node as message
        return src_feat['h'] * edge_feat['e']

    # define reduce function
    def recv_func(feat):
        # we sum the feature vector of the source node
        return fluid.layers.sequence_pool(feat, pool_type='sum')

    # trigger message to passing
    msg = gw.send(send_func, nfeat_list=[('h', nfeat)], efeat_list=[('e', efeat)])
    # recv funciton receives message and trigger reduce funcition to handle message 
    output = gw.recv(msg, recv_func)
    output = fluid.layers.fc(output,
                    size=hidden_size,
                    bias_attr=False,
                    act=activation,
                    name=name)
    return output
```
After defining the GCN layer, we can construct a deeper GCN model with two GCN layers.
```python
output = gcn_layer(gw, gw.node_feat['feature'], gw.edge_feat['edge_feature'],
                hidden_size=8, name='gcn_layer_1', activation='relu')
output = gcn_layer(gw, output, gw.edge_feat['edge_feature'],
                hidden_size=1, name='gcn_layer_2', activation=None)
```

## Step 3:  data preprocessing
Since we implement a node binary classifier, we can use 0 and 1 to represent two classes respectively.
```python 
y = [0,1,1,1,0,0,0,1,0,1]
label = np.array(y, dtype="float32")
label = np.expand_dims(label, -1)
```

## Step 4:  training program
The training process of GCN is the same as that of other paddle-based models.

- First we create a loss function. 
- Then we create a optimizer.
- Finally, we create a executor and train the model. 

```python
# create a label layer as a container 
node_label = fluid.layers.data("node_label", shape=[None, 1],
            dtype="float32", append_batch_size=False)

# using cross-entropy with sigmoid layer as the loss function
loss = fluid.layers.sigmoid_cross_entropy_with_logits(x=output, label=node_label)

# calculate the mean loss
loss = fluid.layers.mean(loss)

# choose the Adam optimizer and set the learning rate to be 0.01
adam = fluid.optimizer.Adam(learning_rate=0.01)
adam.minimize(loss)

# create the executor 
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
feed_dict = gw.to_feed(g) # gets graph data

for epoch in range(30):
    feed_dict['node_label'] = label
    
    train_loss = exe.run(fluid.default_main_program(),
        feed=feed_dict,
        fetch_list=[loss],
        return_numpy=True)
    print('Epoch %d | Loss: %f'%(epoch, train_loss[0]))
```
