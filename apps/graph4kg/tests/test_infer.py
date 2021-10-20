# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from tqdm import tqdm
import paddle
import numpy as np

from train import partial_evaluate


class InferMetricsTest(unittest.TestCase):
    def test_metrics(self):
        np.random.seed(0)
        pos_score = np.random.random([16, 1])
        neg_score = np.random.random([16, 14951])
        for x in range(16):
            neg_score[x][0] = pos_score[x]
        corr_idxs = paddle.zeros((16, 1), dtype='int64')
        pos_score = paddle.to_tensor(pos_score)
        neg_score = paddle.to_tensor(neg_score)

        metrics = partial_evaluate(neg_score, corr_idxs, None)
        mrr = np.sum([x['MRR'] for x in metrics])
        mr = np.sum([x['MR'] for x in metrics])
        self.assertAlmostEqual(mr, 102927.0, None, 'mr not aligned!', 1)
        self.assertAlmostEqual(mrr, 0.00534505, None, 'mrr not aligned!', 1e-5)

    def test_transe_sgpu(self):
        pos_score = np.load('pos_scores.npy')
        neg_score = np.load('neg_scores.npy')
        # print(pos_score[:5])
        # print(neg_score[:5])
        index = []
        for i in tqdm(range(pos_score.shape[0])):
            min_abs = 10000
            min_idx = 0
            for j in range(neg_score[i].shape[0]):
                if abs(pos_score[i] - neg_score[i][j]) < min_abs:
                    min_abs = abs(pos_score[i] - neg_score[i][j])
                    min_idx = j
                    # print(i, min_abs, min_idx)
            index.append(min_idx)
        # print(index, len(index))
        pos_score = paddle.to_tensor(pos_score)
        neg_score = paddle.to_tensor(neg_score)
        index = paddle.to_tensor(index)

        metrics = partial_evaluate(neg_score, index, None)

        mrr = np.sum([x['MRR'] for x in metrics])
        mr = np.sum([x['MR'] for x in metrics])

        self.assertAlmostEqual(mrr, 0.190383, None, 'mrr not aligned!',
                               0.00001)  # 0.649938
        self.assertAlmostEqual(mr, 188.037, None, 'mr not aligned!',
                               0.001)  # 46.6241
        # self.assertAlmostEqual(metrics['average']['HITS@1'],  0.0409337, 0.00001) # 0.526171
        # self.assertAlmostEqual(metrics['average']['HITS@3'], 0.251028, 0.00001) # 0.745805
        # self.assertAlmostEqual(metrics['average']['HITS@10'], 0.509370, 0.00001) # 0.844568

    def test_rotate_sgpu(self):
        pass
        # self.assertAlmostEqual(metrics['average']['MRR'], 0.730356, 0.00001)
        # self.assertAlmostEqual(metrics['average']['MR'], 42.3373, 0.001)
        # self.assertAlmostEqual(metrics['average']['HITS@1'], 0.636606, 0.00001)
        # self.assertAlmostEqual(metrics['average']['HITS@3'], 0.802026, 0.00001)
        # self.assertAlmostEqual(metrics['average']['HITS@10'], 0.875344, 0.00001)
        # training takes 126.15730714797974 seconds                                                 
        # Save model to ckpts/TransE_l2_FB15k_27                                                     
        # [0]Test average MRR: 0.18987879240471772                                                   
        # [0]Test average MR: 187.9824109969359                                                
        # [0]Test average HITS@1: 0.0406713954393865                                                          
        # [0]Test average HITS@3: 0.2504189873203433                                     
        # [0]Test average HITS@10: 0.5098525503208003                                              
        # testing takes 347.130 seconds 
        # 
        # CPU + GPU
        # proc 0 takes 199.458 seconds   
        # training takes 204.3761601448059 seconds 
        # Save model to ckpts/TransE_l2_FB15k_28   
        # [0]Test average MRR: 0.6491943243481275  
        # [0]Test average MR: 40.9613515938447      
        # [0]Test average HITS@1: 0.5232601445717865
        # [0]Test average HITS@3: 0.747769633153324 
        # [0]Test average HITS@10: 0.8481065158876606
        # testing takes 353.857 seconds   

        # 4 * GPU
        # training takes 43.98578763008118 seconds 
        # Save model to ckpts/TransE_l2_FB15k_29   
        # -------------- Test result --------------
        # Test average MRR : 0.5012499192733465    
        # Test average MR : 62.87728326928611      
        # Test average HITS@1 : 0.3212405410438286 
        # Test average HITS@3 : 0.642777335748506  
        # Test average HITS@10 : 0.7782499026595114
        # -----------------------------------------
        # testing takes 80.256 seconds    

        # GPU
        # training takes 1342.265385389328 seconds                                                   
        # Save model to ckpts/RotatE_FB15k_3                                                         
        # [0]Test average MRR: 0.7323492083438987                                                    
        # [0]Test average MR: 38.90190618069781                                                      
        # [0]Test average HITS@1: 0.6395693318210289                                                             
        # [0]Test average HITS@3: 0.8034483926122802                                                 
        # [0]Test average HITS@10: 0.8748793824380829                                                
        # testing takes 361.794 seconds  

        # CPU + GPU
        # training takes 1483.0856294631958 seconds 
        # Save model to ckpts/RotatE_FB15k_4        
        # [0]Test average MRR: 0.7298879195722012   
        # [0]Test average MR: 39.04652028914357     
        # [0]Test average HITS@1: 0.6354048517885257
        # [0]Test average HITS@3: 0.8025934891909736
        # [0]Test average HITS@10: 0.8745915931675441
        # testing takes 384.115 seconds 

        # dglke_train --model_name RotatE --dataset FB15k --batch_size 1024 \
        # --log_interval 1000 --neg_sample_size 256 --regularization_coef 1e-07 \
        # --hidden_dim 200 --gamma 12.0 --lr 0.009 --batch_size_eval 16 --test \
        # -adv -de --max_step 2500 --neg_deg_sample --mix_cpu_gpu --num_proc 4 \
        # --gpu 0 1 2 3 --async_update --rel_part --force_sync_interval 1000 
        # 4 * GPU
        # Test average MRR : 0.43380150801175926
        # Test average MR : 120.30564066970256  
        # Test average HITS@1 : 0.3135379458617596
        # Test average HITS@3 : 0.5006348292732474
        # Test average HITS@10 : 0.6531631426588342
        # testing takes 89.749 seconds 


if __name__ == '__main__':
    unittest.main()
