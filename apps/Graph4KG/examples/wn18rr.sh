DATA_PATH=./data

## Single GPU

# TransE
python -u train.py --model_name TransE --data_name WN18RR --data_path $DATA_PATH --save_path output/transe_wnrr_sgpu \
--batch_size 512 --neg_sample_size 1024 --neg_sample_type 'chunk' --embed_dim 500 --gamma 6.0 -adv --lr 0.1 --optimizer adagrad \
--num_workers 4 --test --print_on_screen --filter_eval -a 0.5 --max_steps 80000

# DistMult
python -u train.py --model_name DistMult --data_name WN18RR --data_path $DATA_PATH --save_path output/distmult_wnrr_sgpu \
--batch_size 512 --neg_sample_type 'chunk' --num_workers 4 --neg_sample_size 1024 --embed_dim 1000 --gamma 200.0 --lr 0.1 --optimizer adagrad \
--test -adv --num_epoch 200 --filter_eval --print_on_screen --reg_coef 5e-6

# ComplEx
python -u train.py --model_name ComplEx --data_name WN18RR --data_path $DATA_PATH --save_path output/complex_wnrr_sgpu \
--batch_size 512 --neg_sample_type 'chunk' --num_workers 4 --neg_sample_size 1024 --embed_dim 500 --gamma 200.0 --lr 0.1 --reg_coef 5e-6 --optimizer adagrad \
--test -adv --num_epoch 200 --filter_eval --print_on_screen -a 1.0

# RotatE
python -u train.py --model_name RotatE --data_name WN18RR --data_path $DATA_PATH --save_path output/rotate_wnrr_sgpu \
--batch_size 512 --reg_coef 1e-7 --neg_sample_size 1024 --neg_sample_type 'chunk' --embed_dim 500 --gamma 6.0 --lr 0.0001 --optimizer adam -adv -a 0.5 \
--num_workers 4 --max_steps 40000 --test --print_on_screen --filter_eval --neg_deg_sample

# OTE
python -u train.py --model_name OTE --data_name WN18RR --data_path $DATA_PATH --save_path output/ote_wnrr_sgpu \
--batch_size 512 --neg_sample_size 256 --neg_sample_type 'chunk' --embed_dim 400 --gamma 5.0 --lr 0.0001 -adv -a 1.8 \
--num_workers 8 --max_steps 50000 --test --print_on_screen --ote_size 4 --ote_scale 2 --optimizer adam --filter_eval

## CPU GPU

# TransE
python -u train.py --model_name TransE --data_name WN18RR --data_path $DATA_PATH --save_path output/transe_wnrr_mgpu \
--batch_size 512 --neg_sample_size 1024 --neg_sample_type 'chunk' --embed_dim 500 --gamma 6.0 -adv --lr 0.1 --optimizer adagrad \
--num_workers 4 --test --print_on_screen --filter_eval -a 0.5 --max_steps 80000 --async_update --mix_cpu_gpu

# DistMult
python -u train.py --model_name DistMult --data_name WN18RR --data_path $DATA_PATH --save_path output/distmult_wnrr_mgpu \
--batch_size 512 --neg_sample_type 'chunk' --num_workers 4 --neg_sample_size 1024 --embed_dim 1000 --gamma 200.0 --lr 0.1 --optimizer adagrad \
--test -adv --num_epoch 200 --filter_eval --print_on_screen --reg_coef 5e-6 --async_update --mix_cpu_gpu

# ComplEx
python -u train.py --model_name ComplEx --data_name WN18RR --data_path $DATA_PATH --save_path output/complex_wnrr_mgpu \
--batch_size 512 --neg_sample_type 'chunk' --num_workers 4 --neg_sample_size 1024 --embed_dim 500 --gamma 200.0 --lr 0.1 --reg_coef 5e-6 --optimizer adagrad \
--test -adv --num_epoch 200 --filter_eval --print_on_screen -a 1.0 --async_update --mix_cpu_gpu

# RotatE
python -u train.py --model_name RotatE --data_name WN18RR --data_path $DATA_PATH --save_path output/rotate_wnrr_mgpu \
--batch_size 512 --reg_coef 1e-7 --neg_sample_size 1024 --neg_sample_type 'chunk' --embed_dim 500 --gamma 6.0 --lr 0.0001 --optimizer adam -adv -a 0.5 \
--num_workers 4 --max_steps 40000 --test --print_on_screen --filter_eval --neg_deg_sample --async_update --mix_cpu_gpu

# OTE
python -u train.py --model_name OTE --data_name WN18RR --data_path $DATA_PATH --save_path output/ote_wnrr_mgpu \
--batch_size 512 --neg_sample_size 256 --neg_sample_type 'chunk' --embed_dim 400 --gamma 5.0 -adv -a 1.8 \
--num_workers 8 --max_steps 10000 --test --print_on_screen --ote_size 4 --ote_scale 2 --lr 0.001 --optimizer adam --async_update --mix_cpu_gpu
