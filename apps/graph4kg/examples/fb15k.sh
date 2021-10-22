# TransE single GPU
python -u train.py --model_name TransE --data_name FB15k --data_path ~/data --save_path output/transe_fb_sgpu \
--batch_size 1000 --test_batch_size 16 --log_interval 1000 --eval_interval 24000 --reg_coef 1e-9 --reg_norm 3 \
--neg_sample_size 200 --neg_sample_type 'chunk' --embed_dim 400 --gamma 19.9 --mlp_lr 0.25 -adv \
--num_workers 2 --num_epoch 50 --valid --test --print_on_screen --filter_eval

# TransE NumPyEmbedding
python -u train.py --model_name TransE --data_name FB15k --data_path ~/data --save_path output/transe_fb_sgpu \
--batch_size 1000 --test_batch_size 16 --log_interval 1000 --eval_interval 24000 --reg_coef 1e-9 --reg_norm 3 \
--neg_sample_size 200 --neg_sample_type 'chunk' --embed_dim 400 --gamma 19.9 --lr 0.25 -adv \
--num_workers 2 --num_epoch 50 --valid --test --print_on_screen --async_update --mix_cpu_gpu --filter_eval

# TransE test
python -u link_prediction.py output/transe_fb_sgpu/TransE_FB15k_d_400_g_19.9

# DistMult single GPU
python -u train.py --model_name DistMult --data_name FB15k --data_path ~/data --save_path output/distmult_fb_sgpu \
--batch_size 1000 --test_batch_size 16 --log_interval 1000 --eval_interval 24000  --neg_sample_type 'chunk' \
--num_workers 2 --neg_sample_size 200 --embed_dim 400 --gamma 143.0 --mlp_lr 0.08 \
--test -adv --num_epoch 50 --filter_eval --print_on_screen --reg_coef 2e-6 --reg_norm 3

# ComplEx single GPU
python -u train.py --model_name ComplEx --data_name FB15k --data_path ~/data --save_path output/complex_fb_sgpu \
--batch_size 1000 --log_interval 1000  --test_batch_size 16 --neg_sample_type 'chunk' --num_workers 2 \
--neg_sample_size 200 --embed_dim 400 --gamma 143.0 --mlp_lr 0.1 --reg_coef 2e-6 \
--test -adv --num_epoch 50 --filter_eval --print_on_screen

# RotatE single GPU
python -u train.py --model_name RotatE --data_name FB15k --data_path ~/data --save_path output/rotate_fb_sgpu \
--batch_size 2048 --test_batch_size 16 --log_interval 1000 --eval_interval 20000 --reg_coef 1e-7 --reg_norm 3 \
--neg_sample_size 256 --neg_sample_type 'chunk' --embed_dim 200 --gamma 12.0 --mlp_lr 0.009 -adv \
--num_workers 2 --num_epoch 100 --test --print_on_screen --ent_times 2 --filter_eval --neg_deg_sample

# RotatE NumPyEmbedding
python -u train.py --model_name RotatE --data_name FB15k --data_path ~/data --save_path output/rotate_fb_sgpu \
--batch_size 2048 --test_batch_size 16 --log_interval 1000 --eval_interval 20000 --reg_coef 1e-7 --reg_norm 3 \
--neg_sample_size 1 --neg_sample_type 'full' --embed_dim 200 --gamma 12.0 --lr 0.009 -adv \
--num_workers 8 --num_epoch 100 --valid --test --print_on_screen --ent_times 2 --async_update --mix_cpu_gpu \
--filter_eval --neg_deg_sample

# RotatE test
python -u link_prediction.py output/rotate_fb_sgpu/RotatE_FB15k_d_200_g_12.0
