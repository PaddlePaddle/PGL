"""
import os
import paddle

    if step == 100:
		paddle.core.nvprof_start()
		paddle.core.nvprof_enable_record_event()
		paddle.core.nvprof_nvtx_push(str(step))
	if step == 110:
		paddle.core.nvprof_nvtx_pop()
		paddle.core.nvprof_stop()
		sys.exit()
	if step > 100 and step < 110:
		paddle.core.nvprof_nvtx_pop()
		paddle.core.nvprof_nvtx_push(str(step))
"""

# TransE single GPU
nsys profile -t cuda,nvtx -o transe_fb15k --capture-range=cudaProfilerApi --stop-on-range-end=true python -u main.py \
--model_name TransE --data_name FB15k --data_path ~/data --save_path output/transe_fb_sgpu \
--batch_size 1000 --reg_coef 1e-9 --reg_norm 3 --neg_sample_size 1 --neg_sample_type 'full' --hidden_dim 400 --gamma 19.9 \
--mlp_lr 0.25 -adv --num_workers 8 --num_epoch 2

# RotatE single GPU
nsys profile -t cuda,nvtx -o rotate_fb15k --capture-range=cudaProfilerApi --stop-on-range-end=true python -u main.py \
--model_name RotatE --data_name FB15k --data_path ~/data --save_path output/rotate_fb_sgpu \
--batch_size 2048 --reg_coef 1e-7 --reg_norm 3 --neg_sample_size 1 --neg_sample_type 'full' --hidden_dim 200 --gamma 12.0 \
--mlp_lr 0.009 -adv --num_workers 8 --num_epoch 2 --ent_times 2 
