## Single GPU

DATA_PATH=../../../data

# TransE
CUDA_VISIBLE_DEVICES=2 python -u train.py --model_name TransE --data_name WN18RR --data_path $DATA_PATH --save_path output/transe_wnrr_sgpu \
--batch_size 512 --neg_sample_size 1024 --neg_sample_type 'chunk' --embed_dim 500 --gamma 6.0 -adv --lr 0.1 --optimizer adagrad \
--num_workers 4 --test --print_on_screen --filter_eval -a 0.5 --max_steps 80000

# DistMult
CUDA_VISIBLE_DEVICES=2 python -u train.py --model_name DistMult --data_name WN18RR --data_path $DATA_PATH --save_path output/distmult_wnrr_sgpu \
--batch_size 512 --neg_sample_type 'chunk' --num_workers 4 --neg_sample_size 1024 --embed_dim 1000 --gamma 200.0 --lr 0.1 --optimizer adagrad \
--test -adv --num_epoch 200 --filter_eval --print_on_screen --reg_coef 5e-6

# ComplEx
CUDA_VISIBLE_DEVICES=2 python -u train.py --model_name ComplEx --data_name WN18RR --data_path $DATA_PATH --save_path output/complex_wnrr_sgpu \
--batch_size 512 --neg_sample_type 'chunk' --num_workers 4 --neg_sample_size 1024 --embed_dim 500 --gamma 200.0 --lr 0.1 --reg_coef 5e-6 --optimizer adagrad \
--test -adv --num_epoch 200 --filter_eval --print_on_screen -a 1.0

# RotatE
CUDA_VISIBLE_DEVICES=2 python -u train.py --model_name RotatE --data_name WN18RR --data_path $DATA_PATH --save_path output/rotate_wnrr_sgpu \
--batch_size 512 --reg_coef 1e-7 --neg_sample_size 1024 --neg_sample_type 'chunk' --embed_dim 500 --gamma 6.0 --lr 0.0001 --optimizer adam -adv -a 0.5 \
--num_workers 4 --max_steps 40000 --test --print_on_screen --filter_eval --neg_deg_sample

# OTE
CUDA_VISIBLE_DEVICES=2 python -u train.py --model_name OTE --data_name WN18RR --data_path $DATA_PATH --save_path output/ote_wnrr_sgpu \
--batch_size 512 --neg_sample_size 256 --neg_sample_type 'chunk' --embed_dim 400 --gamma 5.0 --lr 0.1 -adv -a 1.8 \
--num_workers 8 --max_steps 25000 --test --print_on_screen --ote_size 4 --ote_scale 2 --scheduler_interval 2500 --lr 0.0001 --optimizer adam
