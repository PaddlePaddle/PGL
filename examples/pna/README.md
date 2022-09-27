# Principal Neighbourhood Aggregation for Graph Nets (PNA)

[Principal Neighbourhood Aggregation for Graph Nets \(PNA\)](https://arxiv.org/abs/2004.05718)  is a graph learning model combining multiple aggregators with degree-scalers.


### Datasets

We perform graph classification experiment to reproduce paper results on [OGB](https://ogb.stanford.edu/). 

### Dependencies

- paddlepaddle >= 2.2.0
- pgl >= 2.2.4

### How to run


```
python main.py --config config.yaml   # train on ogbg-molhiv
python main.py --config config_pcba.yaml # train on ogbg-molpcba
```


### Important Hyperparameters

- aggregators: a list of aggregators name. ("mean", "sum", "max", "min", "var", "std")
- scalers: a list of scalers name. ("identity", "amplification", "attenuation", "linear", "inverse_linear")
- tower: The number of towers.
- divide_input: hether the input features should be split between towers or not.
- pre_layers: the number of MLP layers behind aggregators.
- post_layers: MLP layers after aggregator.

### Experiment results （ROC-AUC）
|   | GIN   | PNA(paper result) | PNA(ours)|
|-------------|----------|------------|-----------------|
|HIV    | 0.7778  | 0.7905   | 0.7929     | 
|PCBA   | 0.2266   | 0.2838   | 0.2801      |
