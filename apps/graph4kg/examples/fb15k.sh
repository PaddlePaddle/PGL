""" Single GPU
"""

# TransE
python -u train.py --model_name TransE --data_name FB15k --data_path ~/data --save_path output/transe_fb_sgpu \
--batch_size 1000 --test_batch_size 16 --log_interval 1000 --eval_interval 24000 --reg_coef 1e-9 --reg_norm 3 \
--neg_sample_size 200 --neg_sample_type 'chunk' --embed_dim 400 --gamma 19.9 --lr 0.25 -adv \
--num_workers 2 --num_epoch 50 --valid --test --print_on_screen --filter_eval

# DistMult
python -u train.py --model_name DistMult --data_name FB15k --data_path ~/data --save_path output/distmult_fb_sgpu \
--batch_size 1000 --test_batch_size 16 --log_interval 1000 --eval_interval 24000  --neg_sample_type 'chunk' \
--num_workers 2 --neg_sample_size 200 --embed_dim 400 --gamma 143.0 --lr 0.08 \
--test -adv --num_epoch 50 --filter_eval --print_on_screen --reg_coef 2e-6 --reg_norm 3

# ComplEx
python -u train.py --model_name ComplEx --data_name FB15k --data_path ~/data --save_path output/complex_fb_sgpu \
--batch_size 1000 --log_interval 1000  --test_batch_size 16 --neg_sample_type 'chunk' --num_workers 2 \
--neg_sample_size 200 --embed_dim 400 --gamma 143.0 --lr 0.1 --reg_coef 2e-6 \
--test -adv --num_epoch 50 --filter_eval --print_on_screen

# RotatE
python -u train.py --model_name RotatE --data_name FB15k --data_path ~/data --save_path output/rotate_fb_sgpu \
--batch_size 2048 --test_batch_size 16 --log_interval 1000 --eval_interval 20000 --reg_coef 1e-7 --reg_norm 3 \
--neg_sample_size 256 --neg_sample_type 'chunk' --embed_dim 200 --gamma 12.0 --lr 0.009 -adv \
--num_workers 2 --num_epoch 100 --test --print_on_screen --ent_times 2 --filter_eval --neg_deg_sample

# QuatE
python -u train.py --model_name QuatE --data_name FB15k --data_path ~/data --save_path output/quate_fb_sgpu \
--batch_size 4096 --log_interval 1 --neg_sample_type 'full' --neg_sample_size 20 \
--embed_dim 800 --reg_coef 0.0025 --num_epoch 5000 --lr 0.1 --print_on_screen --reg_type 'norm_er' \
--filter_eval --loss_type Softplus --test --valid --eval_interval 20000 --gamma 8.0

python -m pdb train.py --model_name QuatE --data_name FB15k --data_path ~/data --save_path output/quate_fb_sgpu \
--batch_size 4096 --log_interval 1 --neg_sample_type 'full' --neg_sample_size 20 \
--embed_dim 800 --reg_coef 0.0025 --num_epoch 1 --lr 0.1 --print_on_screen --reg_type 'norm_er' \
--filter_eval --loss_type Softplus --test --valid --eval_interval 20000 --gamma 8.0

""" Multi-GPU
"""
# TransE
python -m paddle.distributed.launch --gpus='1,2' train.py --model_name TransE --data_name FB15k --data_path ~/data --save_path output/transe_fb_sgpu \
--batch_size 1000 --test_batch_size 16 --log_interval 1000 --eval_interval 24000 --reg_coef 1e-9 --reg_norm 3 \
--neg_sample_size 200 --neg_sample_type 'chunk' --embed_dim 400 --gamma 19.9 --lr 0.25 -adv \
--num_workers 8 --num_epoch 50 --valid --test --print_on_screen --async_update --mix_cpu_gpu --filter_eval

# DistMult
python -m paddle.distributed.launch --gpus='0,3' train.py --model_name DistMult --data_name FB15k --data_path ~/data --save_path output/distmult_fb_sgpu \
--batch_size 1000 --test_batch_size 16 --log_interval 1000 --eval_interval 24000  --neg_sample_type 'chunk' \
--num_workers 2 --neg_sample_size 200 --embed_dim 400 --gamma 143.0 --lr 0.08 \
--test -adv --num_epoch 50 --filter_eval --print_on_screen --reg_coef 2e-6 --reg_norm 3 --task_name 'mix-mgpu'

# ComplEx
python -m paddle.distributed.launch --gpus='2,3' train.py --model_name ComplEx --data_name FB15k --data_path ~/data --save_path output/complex_fb_sgpu \
--batch_size 1000 --log_interval 1000  --test_batch_size 16 --neg_sample_type 'chunk' --num_workers 2 \
--neg_sample_size 200 --embed_dim 400 --gamma 143.0 --lr 0.1 --reg_coef 2e-6 \
--test -adv --num_epoch 50 --filter_eval --print_on_screen --mix_cpu_gpu --async_update

# RotatE
python -m paddle.distributed.launch --gpus='2,3' train.py --model_name RotatE --data_name FB15k --data_path ~/data --save_path output/rotate_fb_mgpu \
--batch_size 2048 --test_batch_size 16 --log_interval 1000 --eval_interval 20000 --reg_coef 1e-7 --reg_norm 3 \
--neg_sample_size 256 --neg_sample_type 'chunk' --embed_dim 200 --gamma 12.0 --lr 0.009 -adv \
--num_workers 2 --num_epoch 100 --test --print_on_screen --filter_eval --neg_deg_sample --mix_cpu_gpu --async_update
