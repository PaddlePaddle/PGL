#CUDA_VISIBLE_DEVICES=2 \
#FLAGS_fraction_of_gpu_memory_to_use=0.01 \
#python main.py \
#    --use_cuda \
#    --model TransE \
#    --optimizer adam \
#    --batch_size=512 \
#    --learning_rate=0.001 \
#    --epoch 100 \
#    --evaluate_per_iteration 20 \
#    --sample_workers 4 \
#    --margin 4 \
##    #--only_evaluate

#CUDA_VISIBLE_DEVICES=2 \
#FLAGS_fraction_of_gpu_memory_to_use=0.01 \
#python main.py \
#    --use_cuda \
#    --model RotatE \
#    --data_dir ./data/WN18 \
#    --optimizer adam \
#    --batch_size=512 \
#    --learning_rate=0.001 \
#    --epoch 100 \
#    --evaluate_per_iteration 100 \
#    --sample_workers 10 \
#    --margin 6 \
#    --neg_times 10 

CUDA_VISIBLE_DEVICES=2 \
FLAGS_fraction_of_gpu_memory_to_use=0.01 \
python main.py \
    --use_cuda \
    --model RotatE \
    --data_dir ./data/FB15k \
    --optimizer adam \
    --batch_size=512 \
    --learning_rate=0.001 \
    --epoch 100 \
    --evaluate_per_iteration 100 \
    --sample_workers 10 \
    --margin 8 \
    --neg_times 10 \
    --neg_mode True
