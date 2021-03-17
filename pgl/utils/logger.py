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
"""Logger setup
"""

import logging

log = logging.getLogger('pgl')
console = logging.StreamHandler()
log.addHandler(console)
formatter = logging.Formatter(
    fmt='[%(levelname)s] %(asctime)s [%(filename)12s:%(lineno)5d]:\t%(message)s'
)
console.setFormatter(formatter)
log.setLevel(logging.DEBUG)
log.propagate = False
