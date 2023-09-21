DATA_PATH=./data

## Single GPU

# TransE
python -u train.py --model_name TransE --data_name wn18 --data_path $DATA_PATH --save_path output/transe_wn_sgpu \
--batch_size 1024 -rc 1e-7 --neg_sample_size 256 --neg_sample_type 'chunk' --embed_dim 512 --gamma 6.0 \
--lr 0.1 --optimizer adagrad -adv --num_workers 2 --num_epoch 400 --test --print_on_screen --filter_eval


# DistMult
python -u train.py --model_name DistMult --data_name wn18 --data_path $DATA_PATH --save_path output/distmult_wn_sgpu \
--batch_size 2048 --neg_sample_type 'chunk' \
--num_workers 2 --neg_sample_size 128 --embed_dim 512 --gamma 20.0 --lr 0.15 --optimizer adagrad \
--test -adv --num_epoch 500 --filter_eval --print_on_screen --reg_coef 1e-6


# ComplEx
python -u train.py --model_name ComplEx --data_name wn18 --data_path $DATA_PATH --save_path output/complex_wn_sgpu \
--batch_size 1024 --log_interval 1000  --test_batch_size 16 --neg_sample_type 'chunk' --num_workers 2 \
--neg_sample_size 1024 --embed_dim 512 --gamma 200.0 --lr 0.1 --optimizer adagrad --reg_coef 1e-5 \
--test -adv --num_epoch 250 --filter_eval --print_on_screen


# RotatE
python -u train.py --model_name RotatE --data_name wn18 --data_path $DATA_PATH --save_path output/rotate_wn_sgpu \
--batch_size 2048 --reg_coef 2e-7 --neg_sample_size 64 --neg_sample_type 'chunk' --embed_dim 256 --gamma 9.0 --lr 0.0001 -adv \
--num_workers 2 --num_epoch 200 --test --print_on_screen  --filter_eval --neg_deg_sample --optimizer adam

# OTE
python -u train.py --model_name OTE --data_name wn18 --data_path $DATA_PATH --save_path output/ote_wn_sgpu \
--batch_size 512  --neg_sample_type 'chunk' --neg_sample_size 256 \
--embed_dim 400 --max_steps 2000 --lr 0.001 --optimizer adam  --print_on_screen \
--filter_eval --test --gamma 5.0 --ote_size 4 --ote_scale 2

## CPU-GPU

# TransE
python -u train.py --model_name TransE --data_name wn18 --data_path $DATA_PATH --save_path output/transe_wn_mgpu \
--batch_size 1024 -rc 1e-7 --neg_sample_size 256 --neg_sample_type 'chunk' --embed_dim 512 --gamma 6.0 \
--lr 0.1 --optimizer adagrad -adv --num_workers 2 --num_epoch 400 --test --print_on_screen --filter_eval  --mix_cpu_gpu --async_update


# DistMult
python -u train.py --model_name DistMult --data_name wn18 --data_path $DATA_PATH --save_path output/distmult_wn_mgpu \
--batch_size 2048 --neg_sample_type 'chunk' \
--num_workers 2 --neg_sample_size 128 --embed_dim 512 --gamma 20.0 --lr 0.15 --optimizer adagrad \
--test -adv --num_epoch 500 --filter_eval --print_on_screen --reg_coef 1e-6  --mix_cpu_gpu --async_update


# ComplEx
python -u train.py --model_name ComplEx --data_name wn18 --data_path $DATA_PATH --save_path output/complex_wn_mgpu \
--batch_size 1024 --log_interval 1000  --test_batch_size 16 --neg_sample_type 'chunk' --num_workers 2 \
--neg_sample_size 1024 --embed_dim 512 --gamma 200.0 --lr 0.1 --optimizer adagrad --reg_coef 1e-5 \
--test -adv --num_epoch 250 --filter_eval --print_on_screen  --mix_cpu_gpu --async_update


# RotatE
python -u train.py --model_name RotatE --data_name wn18 --data_path $DATA_PATH --save_path output/rotate_wn_mgpu --lr 0.0001 \
--batch_size 2048 --reg_coef 2e-7 --neg_sample_size 64 --neg_sample_type 'chunk' --embed_dim 256 --gamma 9.0 --cpu_lr 0.0025 -adv \
--num_workers 2 --num_epoch 200 --test --print_on_screen  --filter_eval --neg_deg_sample --optimizer adam  --mix_cpu_gpu --async_update

# OTE
python -u train.py --model_name OTE --data_name wn18 --data_path $DATA_PATH --save_path output/ote_wn_mgpu \
--batch_size 512  --neg_sample_type 'chunk' --neg_sample_size 256 \
--embed_dim 400 --max_steps 3000 --lr 0.001 --optimizer adam --cpu_lr 0.1  --print_on_screen \
--filter_eval --test --gamma 5.0 --ote_size 4 --ote_scale 2 --mix_cpu_gpu --async_update
