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
import os
import json

import numpy as np
import pgl
import paddle.fluid as F
import paddle.fluid.layers as L
from pgl.utils.logger import log
from ernie.file_utils import _fetch_from_remote

from models.encoder import Encoder
from models.loss import Loss


class PretrainedModelLoader(object):
    bce = 'https://ernie-github.cdn.bcebos.com/'
    resource_map = {
        'ernie-1.0': bce + 'model-ernie1.0.1.tar.gz',
        'ernie-2.0-en': bce + 'model-ernie2.0-en.1.tar.gz',
        'ernie-2.0-large-en': bce + 'model-ernie2.0-large-en.1.tar.gz',
        'ernie-tiny': bce + 'model-ernie_tiny.1.tar.gz',
    }

    @classmethod
    def from_pretrained(cls,
                        pretrain_dir_or_url,
                        force_download=False,
                        **kwargs):
        if pretrain_dir_or_url in cls.resource_map:
            url = cls.resource_map[pretrain_dir_or_url]
            log.info('get pretrain dir from %s' % url)
            pretrain_dir = _fetch_from_remote(url, force_download)
        else:
            log.info('pretrain dir %s not in %s, read from local' %
                     (pretrain_dir_or_url, repr(cls.resource_map)))
            pretrain_dir = pretrain_dir_or_url

        if not os.path.exists(pretrain_dir):
            raise ValueError('pretrain dir not found: %s' % pretrain_dir)
        param_path = os.path.join(pretrain_dir, 'params')
        state_dict_path = os.path.join(pretrain_dir, 'saved_weights')
        config_path = os.path.join(pretrain_dir, 'ernie_config.json')

        if not os.path.exists(config_path):
            raise ValueError('config path not found: %s' % config_path)
        name_prefix = kwargs.pop('name', None)
        cfg_dict = dict(json.loads(open(config_path).read()), **kwargs)
        #model = cls(cfg_dict, name=name_prefix)
        log.info('loading pretrained model from %s' % pretrain_dir)
        return cfg_dict, param_path
