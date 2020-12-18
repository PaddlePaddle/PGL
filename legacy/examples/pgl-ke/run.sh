device=3

CUDA_VISIBLE_DEVICES=$device \
FLAGS_fraction_of_gpu_memory_to_use=0.01 \
python main.py \
    --use_cuda \
    --model TransE \
    --data_dir ./data/FB15k \
    --optimizer adam \
    --batch_size=1024 \
    --learning_rate=0.001 \
    --epoch 200 \
    --evaluate_per_iteration 200 \
    --sample_workers 1 \
    --margin 1.0 \
    --nofilter True \
    --neg_times 10 \
    --neg_mode True
    #--only_evaluate

#  TransE FB15k
#  -----Raw-Average-Results
#  MeanRank: 214.94, MRR: 0.2051, Hits@1: 0.0929, Hits@3: 0.2343, Hits@10: 0.4458
#  -----Filter-Average-Results
#  MeanRank:  74.41, MRR: 0.3793, Hits@1: 0.2351, Hits@3: 0.4538, Hits@10: 0.6570



CUDA_VISIBLE_DEVICES=$device \
FLAGS_fraction_of_gpu_memory_to_use=0.01 \
python main.py \
    --use_cuda \
    --model TransE \
    --data_dir ./data/WN18 \
    --optimizer adam \
    --batch_size=1024 \
    --learning_rate=0.001 \
    --epoch 100 \
    --evaluate_per_iteration 100 \
    --sample_workers 1 \
    --margin 4 \
    --nofilter True \
    --neg_times 10 \
    --neg_mode True

#  TransE WN18
#  -----Raw-Average-Results
#  MeanRank: 219.08, MRR: 0.3383, Hits@1: 0.0821, Hits@3: 0.5233, Hits@10: 0.7997
#  -----Filter-Average-Results
#  MeanRank: 207.72, MRR: 0.4631, Hits@1: 0.1349, Hits@3: 0.7708, Hits@10: 0.9315



#for  prertrain
CUDA_VISIBLE_DEVICES=$device \
FLAGS_fraction_of_gpu_memory_to_use=0.01 \
python main.py \
    --use_cuda \
    --model TransE \
    --data_dir ./data/FB15k \
    --optimizer adam \
    --batch_size=512 \
    --learning_rate=0.001 \
    --epoch 30 \
    --evaluate_per_iteration 30 \
    --sample_workers 1 \
    --margin 2.0 \
    --nofilter True \
    --noeval True \
    --neg_times 10 \
    --neg_mode True && \
CUDA_VISIBLE_DEVICES=$device \
FLAGS_fraction_of_gpu_memory_to_use=0.01 \
python main.py \
    --use_cuda \
    --model TransR \
    --data_dir ./data/FB15k \
    --optimizer adam \
    --batch_size=512 \
    --learning_rate=0.001 \
    --epoch 200 \
    --evaluate_per_iteration 200 \
    --sample_workers 1 \
    --margin 2.0 \
    --pretrain True \
    --nofilter True \
    --neg_times 10 \
    --neg_mode True

#  FB15k TransR 200, pretrain 20
#  -----Raw-Average-Results
#  MeanRank: 303.81, MRR: 0.1931, Hits@1: 0.0920, Hits@3: 0.2109, Hits@10: 0.4181
#  -----Filter-Average-Results
#  MeanRank: 156.30, MRR: 0.3663, Hits@1: 0.2318, Hits@3: 0.4352, Hits@10: 0.6231



# for pretrain
CUDA_VISIBLE_DEVICES=$device \
FLAGS_fraction_of_gpu_memory_to_use=0.01 \
python main.py \
    --use_cuda \
    --model TransE \
    --data_dir ./data/WN18 \
    --optimizer adam \
    --batch_size=512 \
    --learning_rate=0.001 \
    --epoch 30 \
    --evaluate_per_iteration 30 \
    --sample_workers 1 \
    --margin 4.0 \
    --nofilter True \
    --noeval True \
    --neg_times 10 \
    --neg_mode True && \
CUDA_VISIBLE_DEVICES=$device \
FLAGS_fraction_of_gpu_memory_to_use=0.01 \
python main.py \
    --use_cuda \
    --model TransR \
    --data_dir ./data/WN18 \
    --optimizer adam \
    --batch_size=512 \
    --learning_rate=0.001 \
    --epoch 100 \
    --evaluate_per_iteration 100 \
    --sample_workers 1 \
    --margin 4.0 \
    --pretrain True \
    --nofilter True \
    --neg_times 10 \
    --neg_mode True

#  TransR WN18 100, pretrain 30
#  -----Raw-Average-Results
#  MeanRank: 321.41, MRR: 0.3706, Hits@1: 0.0955, Hits@3: 0.5906, Hits@10: 0.8099
#  -----Filter-Average-Results
#  MeanRank: 309.15, MRR: 0.5126, Hits@1: 0.1584, Hits@3: 0.8601, Hits@10: 0.9409



CUDA_VISIBLE_DEVICES=$device \
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

# RotatE FB15k
# -----Raw-Average-Results
# MeanRank: 156.85, MRR: 0.2699, Hits@1: 0.1615, Hits@3: 0.3031, Hits@10: 0.5006
# -----Filter-Average-Results
# MeanRank:  53.35, MRR: 0.4776, Hits@1: 0.3537, Hits@3: 0.5473, Hits@10: 0.7062



CUDA_VISIBLE_DEVICES=$device \
FLAGS_fraction_of_gpu_memory_to_use=0.01 \
python main.py \
    --use_cuda \
    --model RotatE \
    --data_dir ./data/WN18 \
    --optimizer adam \
    --batch_size=512 \
    --learning_rate=0.001 \
    --epoch 100 \
    --evaluate_per_iteration 100 \
    --sample_workers 10 \
    --margin 6 \
    --neg_times 10 \
    --neg_mode True

# RotaE WN18
# -----Raw-Average-Results
# MeanRank: 167.27, MRR: 0.6025, Hits@1: 0.4764, Hits@3: 0.6880, Hits@10: 0.8298
# -----Filter-Average-Results
# MeanRank: 155.23, MRR: 0.9145, Hits@1: 0.8843, Hits@3: 0.9412, Hits@10: 0.9570
