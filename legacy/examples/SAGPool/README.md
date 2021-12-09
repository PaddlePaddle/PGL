# Self-Attention Graph Pooling

SAGPool is a graph pooling method based on self-attention. Self-attention uses graph convolution, which allows the pooling method to consider both node features and graph topology. Based on PGL, we implement the SAGPool algorithm and train the model on five datasets.

## Datasets

There are five datasets, including D&D, PROTEINS, NCI1, NCI109 and FRANKENSTEIN. You can download the datasets from [here](https://baidu-pgl.gz.bcebos.com/pgl-data/SAGPool_data.zip), and unzip it directly. The pkl format datasets should be in directory ./data.

## Dependencies

- [paddlepaddle >= 1.8](https://github.com/PaddlePaddle/paddle)
- [pgl 1.1](https://github.com/PaddlePaddle/PGL)

## How to run

```
python main.py --dataset_name DD --learning_rate 0.005 --weight_decay 0.00001

python main.py --dataset_name PROTEINS --learning_rate 0.001 --hidden_size 32 --weight_decay 0.00001

python main.py --dataset_name NCI1 --learning_rate 0.001 --weight_decay 0.00001

python main.py --dataset_name NCI109 --learning_rate 0.0005 --hidden_size 64 --weight_decay 0.0001 --patience 200 

python main.py --dataset_name FRANKENSTEIN --learning_rate 0.001 --weight_decay 0.0001
```

## Hyperparameters

- seed: random seed
- batch\_size: the number of batch size
- learning\_rate: learning rate of optimizer
- weight\_decay: the weight decay for L2 regularization
- hidden\_size: the hidden size of gcn
- pooling\_ratio: the pooling ratio of SAGPool
- dropout\_ratio: the number of dropout ratio
- dataset\_name: the name of datasets, including DD, PROTEINS, NCI1, NCI109, FRANKENSTEIN
- epochs: maximum number of epochs
- patience: patience for early stopping
- use\_cuda: whether to use cuda
- save\_model: the name for the best model

## Performance

We evaluate the implemented method for 20 random seeds using 10-fold cross validation, following the same training procedures as in the paper.

| dataset      | mean accuracy | standard deviation | mean accuracy(paper) | standard deviation(paper) |
| ------------ | ------------- | ------------------ | -------------------- | ------------------------- |
| DD           | 74.4181       | 1.0244             | 76.19                | 0.94                      |
| PROTEINS     | 72.7858       | 0.6617             | 70.04                | 1.47                      |
| NCI1         | 75.781        | 1.2125             | 74.18                | 1.2                       |
| NCI109       | 74.3156       | 1.3                | 74.06                | 0.78                      |
| FRANKENSTEIN | 60.7826       | 0.629              | 62.57                | 0.6                       |
