# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
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
"""Implementation of some helper functions"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os
import time
import yaml
import numpy as np

from pgl.utils.logger import log


class AttrDict(dict):
    """Attr dict
    """

    def __init__(self, d):
        self.dict = d

    def __getattr__(self, attr):
        value = self.dict[attr]
        if isinstance(value, dict):
            return AttrDict(value)
        else:
            return value

    def __str__(self):
        return str(self.dict)


def load_config(config_file):
    """Load config file"""
    with open(config_file) as f:
        if hasattr(yaml, 'FullLoader'):
            config = yaml.load(f, Loader=yaml.FullLoader)
        else:
            config = yaml.load(f)

    return AttrDict(config)
