# Convert PyG model into PGL

This is a demo for converting PyG code into PGL.

```
python gcn.py
```

## PyG GCN

```python
class PyGGCN(torch.nn.Module):
   def __init__(self, input_size, num_class=1, hidden_size=64):
       super(PyGGCN, self).__init__()
       self.conv1 = GCNConv(input_size, hidden_size)
       self.conv2 = GCNConv(hidden_size, num_class)

   def reset_parameters(self):
       self.conv1.reset_parameters()
       self.conv2.reset_parameters()

   def forward(self, x, edges):
       x, edge_index = x, edges.T
       x = F.relu(self.conv1(x, edge_index))
       x = self.conv2(x, edge_index)
       return x
```


## PGL GCN


```python
class PGLGCN(paddle.nn.Layer):
    def __init__(self, input_size, num_class, hidden_size=64):
        super(PGLGCN, self).__init__()
        self.conv1 = pgl.nn.GCNConv(input_size, hidden_size)
        self.conv2 = pgl.nn.GCNConv(hidden_size, num_class)

    def forward(self, x, edges):
        x, edge_index = x, edges
        g = pgl.Graph(num_nodes=x.shape[0], edges=edges)
        x = paddle.nn.functional.relu(self.conv1(g, x))
        x = self.conv2(g, x)
        return x
```

 
