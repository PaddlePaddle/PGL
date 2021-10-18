"""
run these scripts while running fb15k.sh
"""

# TransE single GPU
python single_gpu_mem.py 'python -u main.py --score TransE --dataset FB15k' 100

# RotatE single GPU
python single_gpu_mem.py 'python -u main.py --score RotatE --dataset FB15k' 100


"""
run these scripts after the traning process finished
"""

# TransE single GPU
python average_speed.py output/transe_fb_sgpu/TransE_FB15k_d_400_g_19.9/train.log  100

# RotatE single GPU
python average_speed.py output/rotate_fb_sgpu/RotatE_FB15k_d_200_g_12.0/train.log  100


"""
DGL-KE
"""

# TransE single GPU
python single_gpu_mem.py 'dglke_train --model_name TransE_l2 --dataset FB15k' 100

# RotatE single GPU
python single_gpu_mem.py 'dglke_train --model_name RotatE --dataset FB15k' 100


# TransE single GPU
python average_speed.py ckpt/train.log  100

# RotatE single GPU
python average_speed.py ckpt/train.log  100
