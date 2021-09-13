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
