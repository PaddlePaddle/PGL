device=0
model='gaan'
lr=0.001
drop=0.5

CUDA_VISIBLE_DEVICES=${device} \
    python -u train.py \
    --use_cuda 1 \
    --num_workers 4 \
    --output_path ./output/model \
    --batch_size 1024 \
    --test_batch_size 512 \
    --epoch 100 \
    --learning_rate ${lr} \
    --full_batch 0 \
    --model ${model} \
    --drop_rate ${drop} \
    --samples 8 8 8 \
    --test_samples 20 20 20 \
    --hidden_size 256
