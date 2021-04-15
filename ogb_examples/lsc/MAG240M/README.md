# Baseline Code for MAG240M using PGL

The code is ported from the R-GAT examples [here](https://github.com/snap-stanford/ogb/tree/master/examples/lsc/mag240m). Please refer to the [OGB-LSC paper](https://arxiv.org/abs/2103.09430) for the detailed setting.

## Installation Requirements

```
ogb>=1.3.0
torch>=1.7.0
paddle>=2.0.0
pgl>=2.1.2
```

## Running Preprocessing Script

```
python dataset/sage_author_x.py ./lsc/dataset/
python dataset/sage_institution_x.py ./lsc/dataset/
python dataset/data_generator_rgnn.py ./lsc/dataset/
```

This will give you the following files:

* `author.npy`: The author features, preprocessed by averaging the neighboring paper features.
* `institution_feat.npy`: The institution features, preprocessed by averaging the neighboring author features.
* `full_feat.npy`: The concatenated author, institution, and paper features.
* `full_edge_symmetric_pgl`: The *homogenized* PGL graph.

Since that will usually take a long time, you can download the full_feat.npy as follow. And only run dataset/data_generator_rgnn.py to get PGL graph.

* [`full_feat.npy`](https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/full_feat.npy)

## Running Training Script

```
sh run_lsc_node_rgat_hetegraph.sh
```

## Performance

| Model       |  Valid ACC | 
| ----------- | ---------------| 
| RGAT        | 0.702         | 
