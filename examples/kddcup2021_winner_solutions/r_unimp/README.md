# R-UNIMP for MAG240M using PGL

The code is ported from the R-UNIMP examples [here](https://github.com/snap-stanford/ogb/tree/master/examples/lsc/mag240m).

## Installation Requirements

```
ogb>=1.3.0
torch>=1.7.0
paddle>=2.0.0
pgl>=2.1.2
```

## Running Preprocessing Script

### Get the features:

```
python dataset/sage_author_x.py
python dataset/sage_institution_x.py
python dataset/sage_author_year.py
python dataset/sage_institution_year.py
python dataset/data_generator_r_unimp_sample.py
```

This will give you the following files:

* `author.npy`: The author features, preprocessed by averaging the neighboring paper features.
* `institution_feat.npy`: The institution features, preprocessed by averaging the neighboring author features.
* `author_year.npy`: The author year, preprocessed by averaging the neighboring paper years.
* `institution_year.npy` The institution years, preprocessed by averaging the neighboring author years.
* `full_feat.npy`: The concatenated author, institution, and paper features.
* `all_feat_year.npy`: The concatenated author, institution, and paper years.
* `paper_to_paper_symmetric_pgl_split`: The *paper_to_paper* PGL graph.
* `paper_to_author_symmetric_pgl_split_src`: The *author_to_paper* PGL graph.
* `paper_to_author_symmetric_pgl_split_dst`: The *paper_to_author* PGL graph.
* `institution_edge_symmetric_pgl_split_src`: The *author_to_institution* PGL graph.
* `institution_edge_symmetric_pgl_split_dst`: The *institution_to_author* PGL graph.

Since that will usually take a long time, you can download the full_feat.npy as follow. And only run dataset/data_generator to get pgl graph.

* [`full_feat.npy`](https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/full_feat.npy)

### Get the m2v embedding:

```
AAA
```

### Get the new validation split:

```
python split_valid.py
```

Then, you will save the new cross validation data in follow dir:
* `./valid_64`

## Running Multi-GPU Training Script

```
run_r_unimp_train.sh
```

## Running Multi-GPU Inferring Script
```
run_r_unimp_infer.sh
```
This will give you R_UNIMP value in the performance table below 

## Running Post Process Script

```
AAA
```
This will give you R_UNIMP_POST value in the performance table below 


## Performance

| Model       |  Valid ACC | 
| ----------- | ---------------| 
| R_UNIMP        | 0.771       | 
| R_UNIMP_POST   | 0.771       | 

