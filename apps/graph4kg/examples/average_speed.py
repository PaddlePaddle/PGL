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

import re
import sys
import numpy as np

if __name__ == '__main__':
    path = sys.argv[1]
    skip = int(sys.argv[2])
    rst = {
        'steps': [],
        'forward': [],
        'backward': [],
        'sample': [],
        'update': []
    }

    ans_p = re.compile(
        r'.*sample: (\d*.\d*), forward: (\d*.\d*), backward: (\d*.\d*), update: (\d*.\d*)'
    )
    step_p = re.compile(r'.* (\d*) steps take (\d*.\d*) seconds')

    with open(path, 'r') as rp:
        for line in rp.readlines():
            ans = ans_p.match(line)
            if ans is not None:
                if skip > 0:
                    skip -= 1
                    continue
                rst['sample'].append(float(ans[1]))
                rst['forward'].append(float(ans[2]))
                rst['backward'].append(float(ans[3]))
                rst['update'].append(float(ans[4]))
            step = step_p.match(line)
            if step is not None:
                rst['steps'].append(float(step[2]) / float(step[1]))

    for k, v in rst.items():
        print(k, 'max:', max(v), 'mean:', np.mean(v))
        if k == 'steps':
            print('steps/s', 'max:', 1 / min(v), 'mean:', 1 / np.mean(v))
