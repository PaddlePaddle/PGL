# TransE single GPU
python -u main.py --score TransE --dataset FB15k --data_path ~/data --save_path output/transe_fb_sgpu \
--batch_size 1000 --test_batch_size 16 --log_interval 1000 --eval_interval 24000 --reg_coef 1e-9 --reg_norm 3 \
--num_negs 1 --neg_mode 'full' --hidden_dim 400 --gamma 19.9 --lr 0.25 -adv \
--num_workers 8 --num_epoch 100 --valid --test --print_on_screen

# TransE NumPyEmbedding
python -u main.py --score TransE --dataset FB15k --data_path ~/data --save_path output/transe_fb_sgpu \
--batch_size 1000 --test_batch_size 16 --log_interval 1000 --eval_interval 24000 --reg_coef 1e-9 --reg_norm 3 \
--num_negs 1 --neg_mode 'full' --hidden_dim 400 --gamma 19.9 --lr 0.25 -adv \
--num_workers 8 --num_epoch 100 --valid --test --print_on_screen --async_update --cpu_emb

# TransE test
python -u link_prediction.py output/transe_fb_sgpu/TransE_FB15k_d_400_g_19.9

# RotatE single GPU
python -u main.py --score RotatE --dataset FB15k --data_path ~/data --save_path output/rotate_fb_sgpu \
--batch_size 2048 --test_batch_size 16 --log_interval 1000 --eval_interval 20000 --reg_coef 1e-7 --reg_norm 3 \
--num_negs 1 --neg_mode 'full' --hidden_dim 200 --gamma 12.0 --lr 0.009 -adv \
--num_workers 8 --num_epoch 100 --valid --test --print_on_screen --ent_times 2 

# RotatE NumPyEmbedding
python -u main.py --score RotatE --dataset FB15k --data_path ~/data --save_path output/rotate_fb_sgpu \
--batch_size 2048 --test_batch_size 16 --log_interval 1000 --eval_interval 20000 --reg_coef 1e-7 --reg_norm 3 \
--num_negs 1 --neg_mode 'full' --hidden_dim 200 --gamma 12.0 --lr 0.009 -adv \
--num_workers 8 --num_epoch 100 --valid --test --print_on_screen --ent_times 2 --async_update --cpu_emb

# RotatE test
python -u link_prediction.py output/rotate_fb_sgpu/RotatE_FB15k_d_200_g_12.0
