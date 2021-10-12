set -xe
CUDA_VISIBLE_DEVICES=0  python -u main.py \
    --score TransE \
    --dataset FB15k \
    --data_path ./data \
    --batch_size 1000 --log_interval 1000 --eval_interval 20000 \
    --num_negs 1 --reg_coef=1e-9 --embed_dim 400 --gamma 19.9 \
    --lr 0.25 --test_batch_size 16 -adv --num_epoch 500 --num_worker 8 --neg_mode 'full' \
    --reg_norm 3 \
    --save_path output/transe_fb_sgpu --print_on_screen --valid --test --cpu_emb --async_update
