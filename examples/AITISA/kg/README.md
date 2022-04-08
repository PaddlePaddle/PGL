# The tools that converting the parameter between PGL-Graph4KG and DGL-KE

## Overview
The Knowledge Representation Learning is a hot field, the knowledge representation algorithms had a set of standardized implementations, we could use the tools that converting the parameter between the different framework, the converted parameter would be loaded and infered in the another framework.

## Requirements
- paddlepaddle-gpu==2.2.2
- pgl==2.2.2
- ogb==1.3.1
- torch==1.11.0
- dgl==0.4.3
- dgl==0.1.0.dev

## Supported models
- [x] TransE
- [x] DistMult
- [x] ComplEx
- [x] RotatE
- [x] OTE


## How to use

### script 
The python script as follows:
```text
kg/

├── convert_tools.py # convert script 
├── README.md # use document
```

### PGL-Graph4KG training script 
```
cd  PGL/apps/Graph4KG

python -u train.py --model_name TransE --data_name FB15k --data_path $DATA_PATH --save_path output/transe_fb_sgpu \
--batch_size 1000 --test_batch_size 16 --log_interval 1000 --eval_interval 24000 --reg_coef 1e-9 --reg_norm 2 \
--neg_sample_size 200 --neg_sample_type 'chunk' --embed_dim 400 --gamma 19.9 -adv \
--num_workers 8 --num_epoch 50 --print_on_screen --filter_eval --lr 0.25 --optimizer adagrad --test --use_dict True --kv_mode vk

```
### Convert the PGL-Graph4KG parameter to DGL-KE parameter

```
python convert_tools.py --init_from_ckpt ./pgl_path --save_path ./dgl_path
```

### Evaluate the data from the converted parameter in the DGL-KE
```
dglke_eval --model_name TransE_l1 --dataset FB15k --hidden_dim 400 --gamma 16.0 --batch_size_eval 16 \
#--gpu 0 --model_path ./dgl_path
```

### Convert the DGL-KE parameter to PGL-Graph4KG parameter

```
python convert_tools.py --init_from_ckpt ./dgl_path  --save_path ./pgl_path --mode dgl2pgl --model_name TransE 
```
