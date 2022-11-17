# Solution to MAG240M of NeurIPS2022-OGB-Challenge

The code is about [Boosting the Speed and Performance in Training Large-scale Heterogeneous Graph for OGB-LSC at NeurIPS 2022](./Boosting_the_Speed_and_Performance_in_Training_Large-scale_Heterogeneous_Graph_for_OGB-LSC_at_NeurIPS_2022.pdf).

## Requirements

```
ogb==1.3.5
torch==1.7.0
paddlepaddle-gpu==2.4.0rc0
pgl==2.2.4
```

## Data Preprocessing

### Preprocess Graph Data

The preprocessing of graph data is the same as that of KDDCup2021. Thus, you can go to the `PGL/examples/kddcup2021/MAG240M/r_unimp` and run the following commands:

```
python dataset/sage_author_x.py
python dataset/sage_institution_x.py
python dataset/sage_author_year.py
python dataset/sage_institution_year.py
python dataset/sage_all_data.py
python dataset/merge_m2v_embed.py
```

This will give you the following files:

* `author.npy`: The author features, preprocessed by averaging the neighboring paper features.
* `institution_feat.npy`: The institution features, preprocessed by averaging the neighboring author features.
* `author_year.npy`: The author year, preprocessed by averaging the neighboring paper years.
* `institution_year.npy` The institution years, preprocessed by averaging the neighboring author years.
* `full_feat.npy`: The concatenated author, institution, and paper features.
* `all_feat_year.npy`: The concatenated author, institution, and paper years.
(https://pan.baidu.com/s/1_0PhbFglsWmYdo9fO1CRGQ), using ```dataset/merge_m2v_embed.py``` to merget them together.
* `paper_to_paper_symmetric_pgl_split`: The *paper_to_paper* PGL graph.
* `paper_to_author_symmetric_pgl_split_src`: The *author_to_paper* PGL graph.
* `paper_to_author_symmetric_pgl_split_dst`: The *paper_to_author* PGL graph.
* `institution_edge_symmetric_pgl_split_src`: The *author_to_institution* PGL graph.
* `institution_edge_symmetric_pgl_split_dst`: The *institution_to_author* PGL graph.

### Get the Metapath2vec embedding:

We get metapath2vec embeddings following https://github.com/PaddlePaddle/PGL/tree/static_stable/examples/metapath2vec 

This will give you the following file:

* `m2v_embed.npy`: The m2v embed. you can get it from [here (password: 0mr0)]


### Get the new validation split:

The validation split is the same as that of KDDCup2021. 
Thus, you can go to the `PGL/examples/kddcup2021/MAG240M/r_unimp` and 
run the following command to get the new validation split.

```
python split_valid.py
```

Then, you will save the new cross validation data in the following directory:
* `./valid_64`

the related configuration for validation dataset in `configs/template.yaml` are as follows:

```
valid_path: "./valid_64"
```

### Make 5-fold cross-validation configuration files

We use 5-fold cross-validation in our experiments. 
Thus we need to generate 5 configuration files for each fold as follows:

```
python make_config_files.py ./configs/template.yaml
```

Then you can find 5 configuration files in `./configs/` directory.

## Running Multi-GPU Training Script

```
sh run_train.sh
```

## Running Multi-GPU Inferring Script
```
sh run_infer.sh
```
This will give you R_UNIMP value in the performance table below 

## Running Post Process Script

1. Construct the coauthor graph

```
# Constructed Co-author Graph

python construct_coauthor_graph.py

```

2. Arange all the validation and test prediction file as following  

```
./result/model1
             \_   all_eval_result.npy  # concatenate all validation output
             \_   test_0.npy           # Prediciton for Fold-0 model 
             \_   test_1.npy           # Prediciton for Fold-1 model 
             \_   test_2.npy           # Prediciton for Fold-2 model 
             \_   test_3.npy           # Prediciton for Fold-3 model 
             \_   test_4.npy           # Prediciton for Fold-4 model 
             \_   valid_0.npy          # validation-id for Fold-0
             \_   valid_1.npy          # validation-id for Fold-1
             \_   valid_2.npy          # validation-id for Fold-2
             \_   valid_3.npy          # validation-id for Fold-3
             \_   valid_4.npy          # validation-id for Fold-4

./result/model2
             \_   all_eval_result.npy  # concatenate all validation output
             \_   test_0.npy           # Prediciton for Fold-0 model 
             \_   test_1.npy           # Prediciton for Fold-1 model 
             \_   test_2.npy           # Prediciton for Fold-2 model 
             \_   test_3.npy           # Prediciton for Fold-3 model 
             \_   test_4.npy           # Prediciton for Fold-4 model 
             \_   valid_0.npy          # validation-id for Fold-0
             \_   valid_1.npy          # validation-id for Fold-1
             \_   valid_2.npy          # validation-id for Fold-2
             \_   valid_3.npy          # validation-id for Fold-3
             \_   valid_4.npy          # validation-id for Fold-4
```

3. Runing Post-Smoothing

```
model_name=model1

# set alpha = 0.8 and smoothing for each fold
python post_smoothing.py 0.8 0 ${model_name} 
python post_smoothing.py 0.8 1 ${model_name} 
python post_smoothing.py 0.8 2 ${model_name}
python post_smoothing.py 0.8 3 ${model_name} 
python post_smoothing.py 0.8 4 ${model_name} 


# merge result and generate ./result/${model_name}_diff0.8/all_eval_result.npy
python merge_result.py ${model_name}_diff0.8

```

4. Run ensemble

```
# This will automatically ensemble results from ./result/ and generate y_pred_mag240m.npz 
python ensemble.py
```


This will give you R_UNIMP_POST value in the performance table below 
