# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
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
""" Log writer setup: interface for training visualization.
"""
import six

LogWriter = None

if six.PY3:
    # We highly recommend using VisualDL (https://github.com/PaddlePaddle/VisualDL)
    # for training visualization in Python 3.
    from visualdl import LogWriter
    LogWriter = LogWriter
elif six.PY2:
    from tensorboardX import SummaryWriter
    LogWriter = SummaryWriter
else:
    raise ValueError("Not running on Python2 or Python3 ?")
