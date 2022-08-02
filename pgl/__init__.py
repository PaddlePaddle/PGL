# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""PGL"""

import os
import sys
import logging
import warnings

import paddle
if not (paddle.__version__ >= '2.2.0' or paddle.__version__ == '0.0.0'):
    warnings.warn(
        "The PaddlePaddle version is out of date. Your version is %s, while we need at least 2.2.0."
        % paddle.__version__)

from pgl import graph
from pgl import graph_kernel
from pgl import math
from pgl import nn
from pgl import message
from pgl import dataset
from pgl import utils
from pgl import sampling
from pgl import partition

from pgl.graph import *
from pgl.bigraph import *
from pgl.heter_graph import *

__version__ = "2.2.4"
