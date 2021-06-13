## train_cv0
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m paddle.distributed.launch --log_dir ./output/model_64_valid_0/ r_unimp_multi_gpu_train.py --conf configs/r_unimp_m2v_64_0.yaml

## train_cv1
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m paddle.distributed.launch --log_dir ./output/model_64_valid_1/ r_unimp_multi_gpu_train.py --conf configs/r_unimp_m2v_64_1.yaml


## train_cv2
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m paddle.distributed.launch --log_dir ./output/model_64_valid_2/ r_unimp_multi_gpu_train.py --conf configs/r_unimp_m2v_64_2.yaml


## train_cv3
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m paddle.distributed.launch --log_dir ./output/model_64_valid_3/ r_unimp_multi_gpu_train.py --conf configs/r_unimp_m2v_64_3.yaml

ll

## train_cv4
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m paddle.distributed.launch --log_dir ./output/model_64_valid_4/ r_unimp_multi_gpu_train.py --conf configs/r_unimp_m2v_64_4.yaml
