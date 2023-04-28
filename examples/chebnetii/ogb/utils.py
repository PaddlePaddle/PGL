# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import math
import random
import numpy as np
import pgl
import paddle


def set_seed(seed):
    paddle.seed(seed)
    np.random.seed(seed)


def cheby(i, x):
    if i == 0:
        return 1
    elif i == 1:
        return x
    else:
        T0 = 1
        T1 = x
        for ii in range(2, i + 1):
            T2 = 2 * x * T1 - T0
            T0, T1 = T1, T2
        return T2
