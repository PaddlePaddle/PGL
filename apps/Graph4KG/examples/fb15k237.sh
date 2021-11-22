DATA_PATH=./data

## single GPU

# TransE
python -u train.py --model_name TransE --data_name FB15k-237 --data_path $DATA_PATH \
--save_path output/transe_fb237_sgpu --batch_size 512 --test_batch_size 16 --reg_coef 1e-9 \
--neg_sample_size 256 --neg_sample_type 'chunk' --embed_dim 200 --gamma 5.0 --lr 0.1 --optimizer adagrad -adv \
--num_workers 4 --num_epoch 50 --test --print_on_screen --filter_eval

# DistMult
python -u train.py --model_name DistMult --data_name FB15k-237 --data_path $DATA_PATH \
--save_path output/distmult_fb237_sgpu --batch_size 512 --test_batch_size 16 --neg_sample_type 'chunk' \
--num_workers 4 --neg_sample_size 256 --embed_dim 1000 --gamma 200.0 --lr 0.1 --optimizer adagrad \
--test -adv --num_epoch 50 --filter_eval --print_on_screen --reg_coef 1e-5

# ComplEx
python -u train.py --model_name ComplEx --data_name FB15k-237 --data_path $DATA_PATH \
--save_path output/complex_fb237_sgpu --batch_size 512 --test_batch_size 16 --neg_sample_type 'chunk' \
--num_workers 4 --neg_sample_size 256 --embed_dim 500 --gamma 200.0 --reg_coef 1e-5 \
--test -adv --num_epoch 50 --filter_eval --print_on_screen --lr 0.1 --optimizer adagrad

# RotatE
python -u train.py --model_name RotatE --data_name FB15k-237 --data_path $DATA_PATH \
--save_path output/rotate_fb237_sgpu --batch_size 512 --test_batch_size 16 --reg_coef 1e-7 \
--neg_sample_size 256 --neg_sample_type 'chunk' --embed_dim 200 --gamma 5.0 -adv -a 1.0 \
--num_workers 4 --num_epoch 150 --test --print_on_screen --filter_eval --neg_deg_sample --lr 0.5 --optimizer adagrad

# OTE
python -u train.py --model_name OTE --data_name FB15k-237 --data_path $DATA_PATH --save_path output/ote_fb237_sgpu \
--batch_size 512 --log_interval 1000 --neg_sample_type 'chunk' --neg_sample_size 256 --max_steps 250000 \
--embed_dim 400 --gamma 15.0 -adv -a 0.5 --ote_scale 2 --ote_size 20 --print_on_screen --test --filter_eval --lr 0.002 --optimizer adam

## Mix CPU-GPU

# TransE
python -u train.py --model_name TransE --data_name FB15k-237 --data_path $DATA_PATH \
--save_path output/transe_fb237_mgpu --batch_size 512 --test_batch_size 16 --reg_coef 1e-9 \
--neg_sample_size 256 --neg_sample_type 'chunk' --embed_dim 200 --gamma 5.0 --lr 0.1 --optimizer adagrad -adv \
--num_workers 4 --num_epoch 50 --test --print_on_screen --filter_eval --async_update --mix_cpu_gpu

# DistMult
python -u train.py --model_name DistMult --data_name FB15k-237 --data_path $DATA_PATH \
--save_path output/distmult_fb237_mgpu --batch_size 512 --test_batch_size 16 --neg_sample_type 'chunk' \
--num_workers 4 --neg_sample_size 256 --embed_dim 1000 --gamma 200.0 --lr 0.1 --optimizer adagrad \
--test -adv --num_epoch 50 --filter_eval --print_on_screen --reg_coef 1e-5 --async_update --mix_cpu_gpu

# ComplEx
python -u train.py --model_name ComplEx --data_name FB15k-237 --data_path $DATA_PATH \
--save_path output/complex_fb237_mgpu --batch_size 512 --test_batch_size 16 --neg_sample_type 'chunk' \
--num_workers 4 --neg_sample_size 256 --embed_dim 500 --gamma 200.0 --reg_coef 1e-5 \
--test -adv --num_epoch 50 --filter_eval --print_on_screen --lr 0.1 --optimizer adagrad --async_update --mix_cpu_gpu

# RotatE
python -u train.py --model_name RotatE --data_name FB15k-237 --data_path $DATA_PATH \
--save_path output/rotate_fb237_mgpu --batch_size 512 --test_batch_size 16 --reg_coef 1e-7 \
--neg_sample_size 256 --neg_sample_type 'chunk' --embed_dim 200 --gamma 5.0 -adv -a 1.0 \
--num_workers 4 --num_epoch 150 --test --print_on_screen --filter_eval --neg_deg_sample --lr 0.5 --optimizer adagrad --async_update --mix_cpu_gpu

# OTE
python -u train.py --model_name OTE --data_name FB15k-237 --data_path $DATA_PATH --save_path output/ote_fb237_mgpu \
--batch_size 512 --log_interval 1000 --neg_sample_type 'chunk' --neg_sample_size 256 --max_steps 250000 \
--embed_dim 400 --gamma 15.0 -adv -a 0.5 --ote_scale 2 --ote_size 20 --print_on_screen --test --filter_eval --lr 0.002 --optimizer adam --async_update --mix_cpu_gpu
