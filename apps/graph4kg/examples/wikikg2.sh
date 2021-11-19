# TransE
python -u train.py --model_name TransE --data_name wikikg2 --data_path ../../../../ --save_path output/transe_wikikg2_sgpu \
--batch_size 512 --test_batch_size 16 --log_interval 100 --reg_coef 1e-9 --reg_norm 3 \
--neg_sample_size 128 --neg_sample_type 'chunk' --embed_dim 500 --gamma 30.0 --lr 0.25 -adv -a 1.0 \
--num_workers 4 --test --print_on_screen --async_update --mix_cpu_gpu --max_steps 250000 --cpu_lr 0.25 --optimizer adagrad --valid --eval_interval 25000

# ComplEx
python -u train.py --model_name ComplEx --data_name wikikg2 --data_path ~/data --save_path output/complex_wikikg2_sgpu \
--batch_size 512 --log_interval 1000  --test_batch_size 16 --neg_sample_type 'chunk' --num_workers 4 \
--neg_sample_size 128 --embed_dim 250 --gamma 143.0 --lr 0.1 -adv -a 1.0 \
--test -adv --num_epoch 50 --print_on_screen --async_update --mix_cpu_gpu --cpu_lr 0.1 --optimizer adagrad

# RotatE
python -u train.py --model_name RotatE --data_name wikikg2 --data_path ~/data --save_path output/rotate_wikikg2_sgpu \
--batch_size 512 --test_batch_size 16 --log_interval 1000 \
--neg_sample_size 128 --neg_sample_type 'chunk' --embed_dim 250 --gamma 5.0 --lr 0.01 -adv -a  \
--num_workers 4 --num_epoch 50 --test --print_on_screen --neg_deg_sample --async_update --mix_cpu_gpu --cpu_lr 0.01 --optimizer adagrad

# DistMult
python -u train.py --model_name DistMult --data_name wikikg2 --data_path ~/data --save_path output/distmult_wikikg2_sgpu \
--batch_size 512 --test_batch_size 16 --log_interval 1000 --neg_sample_type 'chunk' \
--num_workers 4 --neg_sample_size 128 --embed_dim 500 --gamma 500.0 --lr 0.1 \
--test -adv --num_epoch 50 --print_on_screen --reg_coef 2e-6 --reg_norm 3  --async_update --mix_cpu_gpu --cpu_lr 0.1 --optimizer adagrad
