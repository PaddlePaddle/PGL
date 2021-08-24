#!/bin/bash

cd ./post_smoothing/rotate/
sh smoothing_run.sh
cd -

hadoop fs -get /post/smoothing/output tmp
python ./post_smoothing/dump_npy.py --data_path tmp \
    --tmp_path tmp.npy 
rm -rf tmp

python ./post_smoothing/get_score.py \
    --tmp_path tmp.npy  \
    --score_func RotatE \
    --output_path ./model_output/RotatE/PostSmoothing/valid_scores.npy \
    --relation_path ./model_output/RotatE/RotatE_wikikg90m_mlp_result/RotatE_relation_mlp.npy \
    --mode "valid" \

python ./post_smoothing/get_score.py \
    --tmp_path tmp.npy  \
    --score_func RotatE \
    --output_path ./model_output/RotatE/PostSmoothing/test_scores.npy \
    --relation_path ./model_output/RotatE/RotatE_wikikg90m_mlp_result/RotatE_relation_mlp.npy \
    --mode "test" \

rm -rf tmp.npy
