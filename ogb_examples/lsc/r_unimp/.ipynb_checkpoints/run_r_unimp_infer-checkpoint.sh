#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m paddle.distributed.launch --log_dir ./output/model_64_valid_0/ lsc_node_hetegraph_unimp_id_split_rd_m2v_fc_all_year_m_infer.py --conf configs/rgat_unimp_init_id_split_rd_path_attn_linear_m2v_fc_label_sample_m2v_64_0.yaml --do_eval
#
#sleep 3

#python  check_cv_get_test_result.py --conf configs/rgat_unimp_init_id_split_rd_path_attn_linear_m2v_fc_label_sample_m2v_64_0.yaml
#sleep 3

#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m paddle.distributed.launch --log_dir ./output/model_64_valid_1/ lsc_node_hetegraph_unimp_id_split_rd_m2v_fc_all_year_m_infer.py --conf configs/rgat_unimp_init_id_split_rd_path_attn_linear_m2v_fc_label_sample_m2v_64_1.yaml --do_eval
#
#sleep 3
#
#python  check_cv_get_test_result.py --conf configs/rgat_unimp_init_id_split_rd_path_attn_linear_m2v_fc_label_sample_m2v_64_1.yaml
#
#sleep 3
#
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m paddle.distributed.launch --log_dir ./output/model_64_valid_2/ lsc_node_hetegraph_unimp_id_split_rd_m2v_fc_all_year_m_infer.py --conf configs/rgat_unimp_init_id_split_rd_path_attn_linear_m2v_fc_label_sample_m2v_64_2.yaml --do_eval
#
#sleep 3
#
#python  check_cv_get_test_result.py --conf configs/rgat_unimp_init_id_split_rd_path_attn_linear_m2v_fc_label_sample_m2v_64_2.yaml
#
#sleep 3
#
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m paddle.distributed.launch --log_dir ./output/model_64_valid_3/ lsc_node_hetegraph_unimp_id_split_rd_m2v_fc_all_year_m_infer.py --conf configs/rgat_unimp_init_id_split_rd_path_attn_linear_m2v_fc_label_sample_m2v_64_3.yaml --do_eval
#
#sleep 3
#
#python  check_cv_get_test_result.py --conf configs/rgat_unimp_init_id_split_rd_path_attn_linear_m2v_fc_label_sample_m2v_64_3.yaml
#
#sleep 3
#
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m paddle.distributed.launch --log_dir ./output/model_64_valid_4/ lsc_node_hetegraph_unimp_id_split_rd_m2v_fc_all_year_m_infer.py --conf configs/rgat_unimp_init_id_split_rd_path_attn_linear_m2v_fc_label_sample_m2v_64_4.yaml --do_eval
#
#sleep 3
#
#python  check_cv_get_test_result.py --conf configs/rgat_unimp_init_id_split_rd_path_attn_linear_m2v_fc_label_sample_m2v_64_4.yaml


python  check_cv_get_test_result.py --conf configs/rgat_unimp_init_id_split_rd_path_attn_linear_m2v_fc_label_sample_m2v_64_4.yaml --eval_all