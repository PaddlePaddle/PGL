DATA_PATH=./data

## GPU

# TransE
python -m paddle.distributed.launch --gpus 0,1,2,3 dist_train.py --model_name TransE --data_name FB15k --data_path $DATA_PATH --save_path output/transe_fb_4gpu \
--batch_size 1000 --test_batch_size 16 --log_interval 1000 --eval_interval 24000 --reg_coef 1e-9 --reg_norm 3 \
--neg_sample_size 200 --neg_sample_type 'chunk' --embed_dim 400 --gamma 19.9 -adv \
--num_workers 2 --max_steps 6000 --print_on_screen --filter_eval --lr 0.25 --optimizer adagrad --cpu_lr 0.25 --cpu_optimizer adagrad --test

# DistMult
python -m paddle.distributed.launch --gpus 0,1,2,3 dist_train.py --model_name DistMult --data_name FB15k --data_path $DATA_PATH --save_path output/distmult_fb_4gpu \
--batch_size 1000 --test_batch_size 16 --neg_sample_type 'chunk' \
--num_workers 2 --neg_sample_size 200 --embed_dim 400 --gamma 143.0 --lr 0.08 --optimizer adagrad \
--test -adv --max_steps 6000 --filter_eval --print_on_screen --reg_coef 2e-6 --reg_norm 3

# ComplEx
python -m paddle.distributed.launch --gpus 0,1,2,3 dist_train.py --model_name ComplEx --data_name FB15k --data_path $DATA_PATH --save_path output/complex_fb_4gpu \
--batch_size 1000 --log_interval 1000  --test_batch_size 16 --neg_sample_type 'chunk' --num_workers 2 \
--neg_sample_size 200 --embed_dim 400 --gamma 143.0 --lr 0.1 --optimizer adagrad --reg_coef 2e-6 \
--test -adv --max_steps 6000 --filter_eval --print_on_screen

# RotatE
python -m paddle.distributed.launch --gpus 0,1,2,3 dist_train.py --model_name RotatE --data_name FB15k --data_path $DATA_PATH --save_path output/rotate_fb_4gpu \
--batch_size 2048 --test_batch_size 16 --log_interval 1000 --reg_coef 1e-7 --reg_norm 3 \
--neg_sample_size 256 --neg_sample_type 'chunk' --embed_dim 200 --gamma 12.0 --lr 0.009 --optimizer adagrad -adv \
--num_workers 2 --max_steps 12000 --test --print_on_screen --filter_eval --neg_deg_sample

# OTE
python -m paddle.distributed.launch --gpus 0,1,2,3 dist_train.py --model_name OTE --data_name FB15k --data_path $DATA_PATH --save_path output/ote_fb_4gpu \
--batch_size 512 --log_interval 1000 --neg_sample_type 'chunk' --neg_sample_size 256 --max_steps 62500 \
--embed_dim 400 --gamma 15.0 -adv -a 0.5 --ote_scale 2 --ote_size 20 --print_on_screen --test --filter_eval --lr 0.002 --optimizer adam --scheduler_interval 6250

## CPU-GPU

# TransE
python -m paddle.distributed.launch --gpus 0,1,2,3 dist_train.py --model_name TransE --data_name FB15k --data_path $DATA_PATH --save_path output/transe_fb_4mgpu \
--batch_size 1000 --test_batch_size 16 --log_interval 1000 --reg_coef 1e-9 --reg_norm 3 \
--neg_sample_size 200 --neg_sample_type 'chunk' --embed_dim 400 --gamma 19.9 -adv --lr 0.25 --optimizer adagrad --cpu_lr 0.25 \
--num_workers 8 --max_steps 6000 --test --print_on_screen --async_update --mix_cpu_gpu --filter_eval

# DistMult
python -m paddle.distributed.launch --gpus 0,1,2,3 dist_train.py --model_name DistMult --data_name FB15k --data_path $DATA_PATH --save_path output/distmult_fb_4mgpu \
--batch_size 1000 --test_batch_size 16 --log_interval 1000 --neg_sample_type 'chunk' \
--num_workers 2 --neg_sample_size 200 --embed_dim 400 --gamma 143.0 --lr 0.08 --optimizer adagrad --cpu_lr 0.1 \
--test -adv --max_steps 6000 --filter_eval --print_on_screen --reg_coef 2e-6 --reg_norm 3 --mix_cpu_gpu --async_update

# ComplEx
python -m paddle.distributed.launch --gpus 0,1,2,3 dist_train.py --model_name ComplEx --data_name FB15k --data_path $DATA_PATH --save_path output/complex_fb_4mgpu \
--batch_size 1000 --log_interval 1000  --test_batch_size 16 --neg_sample_type 'chunk' --num_workers 2 \
--neg_sample_size 200 --embed_dim 400 --gamma 143.0 --lr 0.1 --optimizer adagrad --reg_coef 2e-6 \
--test -adv --max_steps 6000 --filter_eval --print_on_screen --mix_cpu_gpu --async_update

# RotatE
python -m paddle.distributed.launch --gpus 0,1,2,3 dist_train.py --model_name RotatE --data_name FB15k --data_path $DATA_PATH --save_path output/rotate_fb_4mgpu \
--batch_size 2048 --test_batch_size 16 --log_interval 1000 --eval_interval 20000 --reg_coef 1e-7 --reg_norm 3 \
--neg_sample_size 256 --neg_sample_type 'chunk' --embed_dim 200 --gamma 12.0 --lr 0.009 --optimizer adam --cpu_lr 0.1 -adv \
--num_workers 2 --max_steps 12000 --test --print_on_screen --filter_eval --neg_deg_sample --mix_cpu_gpu --async_update

# OTE
python -m paddle.distributed.launch --gpus 0,1,2,3 dist_train.py --model_name OTE --data_name FB15k --data_path $DATA_PATH --save_path output/ote_fb_4mgpu \
--batch_size 512 --log_interval 1000 --neg_sample_type 'chunk' --neg_sample_size 256 --max_steps 62500 --async_update --mix_cpu_gpu \
--embed_dim 400 --gamma 15.0 -adv -a 0.5 --ote_scale 2 --ote_size 20 --print_on_screen --test --filter_eval --lr 0.002 --optimizer adam --scheduler_interval 6250
