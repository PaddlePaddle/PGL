#!/bin/bash

cd ./post_smoothing/transe/
sh smoothing_run.sh
cd -

hadoop fs -get /post/smoothing/output tmp
python ./post_smoothing/dump_npy.py --data_path tmp \
    --tmp_path tmp.npy 
rm -rf tmp

python ./post_smoothing/get_score.py \
    --tmp_path tmp.npy  \
    --score_func TransE \
    --output_path ./model_output/TransE/PostSmoothing/valid_scores.npy \
    --relation_path ./model_output/TransE/TransE_l2_wikikg90m_mlp_result/TransE_l2_relation_mlp.npy \
    --mode "valid" \

python ./post_smoothing/get_score.py \
    --tmp_path tmp.npy  \
    --score_func TransE \
    --output_path ./model_output/TransE/PostSmoothing/test_scores.npy \
    --relation_path ./model_output/TransE/TransE_l2_wikikg90m_mlp_result/TransE_l2_relation_mlp.npy \
    --mode "test" \

rm -rf tmp.npy
