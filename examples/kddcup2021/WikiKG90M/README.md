# NOTE for WikiKG90M using PGL

The code is about [《NOTE: Solution for KDD-Cup 2021 WikiKG90M-LSC》](./NOTE__SOLUTION_FOR_KDD_CUP_2021_WikiKG90M_LSC.pdf). 

## Installation Requirements

```
ogb>=1.3.0
torch>=1.7.0
paddle>=2.0.0
pgl>=2.1.2
```

## Running Manual Features

### Get the features:

```
    python ./feature/walk_probability/h2r.py
    python ./feature/walk_probability/h2t.py
    python ./feature/walk_probability/r2t.py
    python ./feature/walk_probability/r2h.py
    python ./feature/walk_probability/t2r.py
    python ./feature/walk_probability/t2h.py

    python ./feature/dump_feat/1_rrt_feat.py
    python ./feature/dump_feat/2_h2t_t2h_feat.py
    python ./feature/dump_feat/3_t2h_h2t_feat.py
    python ./feature/dump_feat/4_h2t_h2t_feat.py
    python ./feature/dump_feat/5_hht_feat_byprob.py
    python ./feature/dump_feat/6_r2t_h2r_feat.py
    python ./feature/dump_feat/7_r2t_feat.py
    python ./feature/dump_feat/8_rrh_feat.py
    python ./feature/dump_feat/9_rt_feat.py
    python ./feature/dump_feat/10_ht_feat.py
```
This will give you the manual features in `feature_output`.

## Running Knowledge Embedding Model

Please refer to [Graph4KG](https://github.com/PaddlePaddle/PGL/tree/main/apps/Graph4KG) for paddle version code.

The DGL-KE version code is in [model_dglke](https://github.com/WeiyueSu/PGL/tree/wikikg90m-dglke/examples/kddcup2021/WikiKG90M/model_dglke).
```
sh ./model_dglke/run_train_all.sh
sh ./model_dglke/run_infer_all.sh
```
This will give you the manual features in `model_output`.

## Running Post Smoothing
For the trained TransE and RotatE models, we use post smooothing technology to further improve the effect
```
sh post_smoothing/run_transe.sh
sh post_smoothing/run_rotate.sh
```

## Running ensamble

```
# This will automatically ensemble results from ./model_output/, ./feature_output/ and generate t_pred_wikikg90m.npz
python ensemble.py
```
This will give you **NOTE** value in the performance table below 

## Performance

| Model       |  Valid ACC | 
| ----------- | ---------------| 
| Final Ensemble(w/ manual feature) | 0.9797      |
