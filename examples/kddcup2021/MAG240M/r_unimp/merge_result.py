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

import sys
import numpy as np
import torch
from ogb.lsc import MAG240MDataset, MAG240MEvaluator

root = "dataset_path"
split = torch.load(os.path.join(root, "mag240_kddcup2021", "split_dict.pt"))
val_idx = split['valid']

C = 153
import sys
model_name = sys.argv[1]

valid_output_merge = np.zeros((len(val_idx), 153), dtype="float32")

val_idx_dict = {}
for n, vid in enumerate(val_idx):
    val_idx_dict[vid] = n

for i in range(5):
    nid = np.load("result/" + model_name + "/valid_%s.npy" % i)
    pred = np.load("result/" + model_name + "/val_%s_pred.npy" % i)
    nid = [val_idx_dict[n] for n in nid]
    valid_output_merge[nid, :] = pred

np.save("result/" + model_name + "/all_eval_result.npy", valid_output_merge)
