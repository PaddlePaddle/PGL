""" Single GPU
"""

# TransE
CUDA_VISIBLE_DEVICES=2 python -u train.py --model_name TransE --data_name WN18RR --data_path ~/data --save_path output/transe_wnrr_sgpu \
--batch_size 512 --test_batch_size 16 --log_interval 1000 --eval_interval 15000 \
--neg_sample_size 1024 --neg_sample_type 'chunk' --embed_dim 500 --gamma 6.0 --lr 5e-5 -adv \
--num_workers 4 --num_epoch 200 --valid --test --print_on_screen --filter_eval -a 0.5

# DistMult
CUDA_VISIBLE_DEVICES=2 python -u train.py --model_name DistMult --data_name WN18RR --data_path ~/data --save_path output/distmult_wnrr_sgpu \
--batch_size 512 --test_batch_size 16 --log_interval 1000 --eval_interval 15000  --neg_sample_type 'chunk' \
--num_workers 4 --neg_sample_size 1024 --embed_dim 1000 --gamma 200.0 --lr 0.1 \
--test -adv --num_epoch 200 --filter_eval --print_on_screen --reg_coef 5e-6 --eval_interval 15000 --valid

# ComplEx
CUDA_VISIBLE_DEVICES=2 python -u train.py --model_name ComplEx --data_name WN18RR --data_path ~/data --save_path output/complex_wnrr_sgpu \
--batch_size 512 --log_interval 1000  --test_batch_size 16 --neg_sample_type 'chunk' --num_workers 4 \
--neg_sample_size 1024 --embed_dim 500 --gamma 200.0 --lr 0.1 --reg_coef 5e-6 \
--test -adv --num_epoch 200 --filter_eval --print_on_screen --eval_interval 15000 --valid -a 1.0

# RotatE
CUDA_VISIBLE_DEVICES=2 python -u train.py --model_name RotatE --data_name WN18RR --data_path ~/data --save_path output/rotate_wnrr_sgpu \
--batch_size 512 --test_batch_size 16 --log_interval 1000 --eval_interval 20000 --reg_coef 1e-7 --reg_norm 3 \
--neg_sample_size 1024 --neg_sample_type 'chunk' --embed_dim 500 --gamma 6.0 --lr 0.1 -adv -a 0.5 \
--num_workers 4 --num_epoch 200 --test --print_on_screen --filter_eval --neg_deg_sample  --eval_interval 15000 --valid

# OTE
CUDA_VISIBLE_DEVICES=1 python -u train.py --model_name OTE --data_name WN18RR --data_path ~/data --save_path output/ote_wnrr_sgpu \
--batch_size 512 --test_batch_size 16 --log_interval 100 --eval_interval 1000 --reg_coef 0 \
--neg_sample_size 256 --neg_sample_type 'chunk' --embed_dim 400 --gamma 5.0 --lr 0.1 -adv -a 1.8 \
--num_workers 8 --num_epoch 500 --test --print_on_screen --filter_eval --eval_interval 15000 --valid --ote_size 4 --ote_scale 2
