DATA_PATH=./data/

# TransE
python train.py \
    --model_name TransE \
    --data_path $DATA_PATH \
    --data_name wikikg90m \
    --embed_dim 100 --gamma 8. --reg_coef 1e-9 \
    -adv --mix_cpu_gpu --num_workers 4 \
    --lr 0.00002 --cpu_lr 0.15 \
    --optimizer adam --valid_percent 0.1 \
    --max_steps 2000000 \
    --batch_size 1000 --neg_sample_size 1000 \
    --neg_sample_type 'chunk' \
    --log_interval 1000 --num_process 1 \
    --async_update --use_feature \
    --print_on_screen --save_path output/wikikg90m_transe/ \
    --test --valid --eval_interval 20000

# RotatE
python train.py \
    --model_name RotatE \
    --data_path $DATA_PATH \
    --data_name wikikg90m \
    --embed_dim 100 --gamma 8. --reg_coef 1e-9 \
    -adv --mix_cpu_gpu --num_workers 4 \
    --lr 0.00002 --cpu_lr 0.15 \
    --optimizer adam --valid_percent 0.1 \
    --max_steps 2000000 \
    --batch_size 1000 --neg_sample_size 1000 \
    --neg_sample_type 'chunk' \
    --log_interval 1000 --num_process 1 \
    --async_update --use_feature \
    --print_on_screen --save_path output/wikikg90m_rotate/ \
    --test --valid --eval_interval 20000

# OTE
python train.py \
    --model_name OTE \
    --data_path $DATA_PATH \
    --data_name wikikg90m \
    --embed_dim 200 --gamma 8. --reg_coef 1e-9 \
    -adv --mix_cpu_gpu --num_workers 4 \
    --lr 0.00002 --cpu_lr 0.15 \
    --optimizer adam --valid_percent 0.1 \
    --max_steps 2000000 \
    --ote_size 20 --ote_scale 2 \
    --batch_size 1000 --neg_sample_size 1000 \
    --neg_sample_type 'chunk' \
    --log_interval 1000 --num_process 1 \
    --async_update --use_feature \
    --print_on_screen --save_path output/wikikg90m_ote/ \
    --test --valid --valid_percent 0.1 --eval_interval 20000
