# N-OTE for WikiKG90M using PGL

The code is about [《N-OTE: Solution for KDD-Cup 2021 WikiKG90M-LSC》](./LSC.pdf). 

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

### Get the Metapath2vec embedding:

We get metapath2vec embeddings following https://github.com/PaddlePaddle/PGL/tree/static_stable/examples/metapath2vec 


## Running Training Script

```
sh ./model/mlplr0.00002_lrdecay_bs1kneg1k_ote.sh
```

## Running Inferring Script
```
sh run_infer.sh
```

3. Run ensemble

```
# This will automatically ensemble results from ./model_output/, ./feature_output/ and generate t_pred_wikikg90m.npz
python ensemble.py
```


This will give you **NOTE** value in the performance table below 


## Performance

| Model       |  Valid ACC | 
| ----------- | ---------------| 
| Final Ensemble(w/ manual feature) | 0.9797      |
