# struc2vec: Learning Node Representations from Structural Identity
[Struc2vec](https://arxiv.org/abs/1704.03165) is is a concept of symmetry in which network nodes are identified according to the network structure and their relationship to other nodes. A novel and flexible framework for learning latent representations is proposed in the paper of struc2vec. We reproduce Struc2vec algorithm in the PGL.
###  DataSet
The paper of use air-traffic network to valid algorithm of Struc2vec.
The each edge in the dataset indicate that having one flight between the airports. Using the the connection between the airports to predict the level of activity. The following dataset will be used to valid the algorithm accuracy.Data collected from the Bureau of Transportation Statistics2 from January to October, 2016. The network has 1,190 nodes, 13,599 edges (diameter is 8). [Link](https://www.transtats.bts.gov/)

- usa-airports.edgelist 
- labels-usa-airports.txt

### Dependencies
If use want to use the struc2vec model in pgl, please install the gensim, pathos, fastdtw additional.
- paddlepaddle>=1.6
- pgl
- gensim 
- pathos
- fastdtw

### How to use
For examples, we want to train and valid the Struc2vec model on American airpot dataset 
> python struc2vec.py --edge_file data/usa-airports.edgelist --label_file data/labels-usa-airports.txt --train True --valid True --opt2 True

### Hyperparameters
| Args| Meaning|
| ------------- | ------------- |
| edge_file | input file name for edges|
| label_file | input file name for node label|
| emb_file | input file name for node label|
| walk_depth| The step3 for random walk|
| opt1| The flag to open optimization 1 to reduce time cost|
| opt2| The flag to open optimization 2 to reduce time cost|
| w2v_emb_size| The dims of output the word2vec embedding|
| w2v_window_size| The context length of word2vec|
| w2v_epoch| The num of epoch to train the model.|
| train| The flag to run the struc2vec algorithm to get the w2v embedding|
| valid| The flag to use the w2v embedding to valid the classification result|
| num_class| The num of class in classification model to be trained|

###  Experiment results
| Dataset | Model | Metric | PGL Result | Paper repo Result |
| ------------- | ------------- |------------- |------------- |------------- |
| American airport dataset | Struc2vec without time cost optimization| ACC |0.6483|0.6340|
| American airport dataset | Struc2vec with optimization 1| ACC |0.6466|0.6242|
| American airport dataset | Struc2vec with optimization 2| ACC |0.6252|0.6241|
| American airport dataset | Struc2vec with optimization1&2| ACC |0.6226|0.6083|
