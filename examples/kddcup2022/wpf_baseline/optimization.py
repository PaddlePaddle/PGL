# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle as P
import re


def get_optimizer(model, learning_rate):
    g_clip = P.nn.ClipGradByNorm(50.0)  #experimental
    opt = P.optimizer.Adam(
        learning_rate=learning_rate,
        parameters=model.parameters(),
        grad_clip=g_clip)
    return opt
