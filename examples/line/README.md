# LINE: Large-scale Information Network Embedding
[LINE](http://www.www2015.it/documents/proceedings/proceedings/p1067.pdf) is an algorithmic framework for embedding very large-scale information networks. It is suitable to a variety of networks including directed, undirected, binary or weighted edges. Based on PGL, we reproduce LINE algorithms and reach the same level of indicators as the paper.

## Datasets
[Flickr network](http://socialnetworks.mpi-sws.org/data-imc2007.html) is a social network, which contains 1715256 nodes and 22613981 edges.

You can dowload data from [here](http://socialnetworks.mpi-sws.org/data-imc2007.html).

Flickr network contains four files: 
* flickr-groupmemberships.txt.gz
* flickr-groups.txt.gz
* flickr-links.txt.gz
* flickr-users.txt.gz

After downloading the data，uncompress them, let's say, in **./data/flickr/** . Note that the current directory is the root directory of LINE model.

Then you can run the below command to preprocess the data.
```sh
python data_process.py
```

Then it will produce three files in **./data/flickr/** directory: 
* nodes.txt
* edges.txt
* nodes_label.txt


## Dependencies
- paddlepaddle>=1.6
- pgl

## How to run

For examples, use gpu to train LINE on Flickr dataset.
```sh
# multiclass task example
python line.py --use_cuda --order first_order --data_path ./data/flickr/ --save_dir ./checkpoints/model/

python multi_class.py --ckpt_path ./checkpoints/model/model_epoch_20 --percent 0.5

```

## Hyperparameters

- -use_cuda: Use gpu if assign use_cuda.
- -order: LINE with First_order Proximity or Second_order Proximity
- -percent: The percentage of data as training data

### Experiment results
Dataset|model|Task|Metric|PGL Result|Reported Result
--|--|--|--|--|--
Flickr|LINE with first_order|multi-label classification|MacroF1|0.626|0.627
Flickr|LINE with first_order|multi-label classification|MicroF1|0.637|0.639
Flickr|LINE with second_order|multi-label classification|MacroF1|0.615|0.621
Flickr|LINE with second_order|multi-label classification|MicroF1|0.630|0.635
