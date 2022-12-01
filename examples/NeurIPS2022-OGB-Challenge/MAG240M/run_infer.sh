# test cv1

for cv in `seq 0 4`; do

    echo "=====================================INFO INFER valid ${cv}===================================================="
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m paddle.distributed.launch --log_dir logs/infer/valid_${cv} r_unimp_infer.py --conf configs/r_unimp_peg_gpr_${cv}.yaml --do_eval
    sleep 3
    python3  check_cv_get_test_result.py --conf configs/r_unimp_peg_gpr_${cv}.yaml

done

echo "=====================================INFO INFER valid ALL===================================================="
python3  check_cv_get_test_result.py --conf configs/r_unimp_peg_gpr_4.yaml --eval_all
