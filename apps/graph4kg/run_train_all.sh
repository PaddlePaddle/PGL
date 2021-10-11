# FB15k
# The TransE model
python -u main.py \
    --score TransE \
    --dataset FB15k \
    --data_path ~/data/ \
    --batch_size 1000 \
    --log_interval 1000 \
    --eval_interval 20000 \
    --num_negs 1 \
    --reg_coef=0 \
    --embed_dim 400 \
    --gamma 19.9 \
    --lr 0.25 \
    --test_batch_size 16 \
    -adv \
    --num_epoch 500 \
    --num_worker 8 \
    --neg_mode 'batch' \
    --save_path output/transe_fb_sgpu \
    --print_on_screen \
    --valid --test \
    --cpu_emb --async_update

# wikikg90m
# The TransE model
python -u  main.py \
    --dataset wikikg90m \
    --cpu_emb \
    --hidden_dim 200 \
    --eval_interval 50000 \
    --eval_percent 0.01 \
    --valid \
    --print_on_screen \
    --log_interval 200 \
    --gamma 8 \
    --lr 0.1 \
    --mlp_lr 2e-5 \
    --max_step 2000000\
    --regularization_coef 1e-9 \
    -adv

# The RotatE model
python -u  main.py \
    --dataset wikikg90m \
    --model RotatE \
    --hidden_dim 100 \
    --cpu_emb \
    --eval_interval 50000 \
    --eval_percent 0.01 \
    --valid \
    --print_on_screen \
    --log_interval 200 \
    --gamma 8 \
    --lr 0.1 \
    --mlp_lr 2e-5\
    --max_step 2000000\
    --regularization_coef 1e-9 \
    -de \
    -adv

# The OTE model
python -u  main.py \
    --dataset wikikg90m \
    --model OTE\
    --hidden_dim 200 \
    --cpu_emb \
    --eval_interval 50000 \
    --eval_percent 0.01 \
    --valid \
    --print_on_screen \
    --log_interval 200 \
    --gamma 12 \
    --lr 0.1 \
    --mlp_lr 2e-5\
    --max_step 4000000\
    --regularization_coef 1e-9 \
    --ote_size 40 \
    -adv
