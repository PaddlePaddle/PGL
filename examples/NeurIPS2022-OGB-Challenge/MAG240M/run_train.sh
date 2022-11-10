echo `which python3`
for cv in `seq 0 4`; do

    echo "=====================================INFO TRAIN valid ${cv}===================================================="
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m paddle.distributed.launch --log_dir ./logs/train/valid_${cv} r_unimp_train_p2p.py --conf configs/r_unimp_peg_gpr_${cv}.yaml --ensemble_setting

done
