# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""test_alias_sample"""
import argparse
import time
import unittest
from collections import Counter

import numpy as np

from pgl.graph_kernel import alias_sample_build_table
from pgl.sample import alias_sample


class AliasSampleTest(unittest.TestCase):
    """AliasSampleTest
    """

    def setUp(self):
        pass

    def test_speed(self):
        """test_speed
        """

        num = 1000
        size = [10240, 1, 5]
        probs = np.random.uniform(0.0, 1.0, [num])
        probs /= np.sum(probs)

        start = time.time()
        alias, events = alias_sample_build_table(probs)
        for i in range(100):
            alias_sample(size, alias, events)
        alias_sample_time = time.time() - start

        start = time.time()
        for i in range(100):
            np.random.choice(num, size, p=probs)
        np_sample_time = time.time() - start
        self.assertTrue(alias_sample_time < np_sample_time)

    def test_resut(self):
        """test_result
        """
        size = [450000]
        num = 10
        probs = np.arange(1, num).astype(np.float64)
        probs /= np.sum(probs)
        alias, events = alias_sample_build_table(probs)
        ret = alias_sample(size, alias, events)
        cnt = Counter(ret)
        sort_cnt_keys = [x[1] for x in sorted(zip(cnt.values(), cnt.keys()))]
        self.assertEqual(sort_cnt_keys, np.arange(0, num - 1).tolist())


if __name__ == '__main__':
    unittest.main()
