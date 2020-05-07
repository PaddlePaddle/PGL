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
"""
This file implement a class for model configure.
"""

import datetime
import os
import yaml
import random
import shutil
import six
import logging

log = logging.getLogger("logger")


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


class Config(object):
    """Implementation of Config class for model configure.

    Args:
        config_file(str): configure filename, which is a yaml file.
        isCreate(bool): if true, create some neccessary directories to save models, log file and other outputs.
        isSave(bool): if true, save config_file in order to record the configure message.
    """

    def __init__(self, config_file, isCreate=False, isSave=False):
        self.config_file = config_file
        #  self.config = self.get_config_from_yaml(config_file)
        self.config = self.load_config(config_file)

        if isCreate:
            self.create_necessary_dirs()

            if isSave:
                self.save_config_file()

    def load_config(self, config_file):
        """Load config file"""
        with open(config_file) as f:
            if hasattr(yaml, 'FullLoader'):
                config = yaml.load(f, Loader=yaml.FullLoader)
            else:
                config = yaml.load(f)
        return config

    def create_necessary_dirs(self):
        """Create some necessary directories to save some important files.
        """

        self.config['log_dir'] = os.path.join(self.config['log_dir'],
                                              self.config['task_name'])
        self.config['save_dir'] = os.path.join(self.config['save_dir'],
                                               self.config['task_name'])
        self.config['output_dir'] = os.path.join(self.config['output_dir'],
                                                 self.config['task_name'])

        self.make_dir(self.config['log_dir'])
        self.make_dir(self.config['save_dir'])
        self.make_dir(self.config['output_dir'])

    def save_config_file(self):
        """Save config file so that we can know the config when we look back
        """
        filename = self.config_file.split('/')[-1]
        targetpath = os.path.join(self.config['save_dir'], filename)
        try:
            shutil.copyfile(self.config_file, targetpath)
        except shutil.SameFileError:
            log.info("%s and %s are the same file, did not copy by shutil"\
                    % (self.config_file, targetpath))

    def make_dir(self, path):
        """Build directory"""
        if not os.path.exists(path):
            os.makedirs(path)

    def __getitem__(self, key):
        return self.config[key]

    def __call__(self):
        """__call__"""
        return self.config

    def __getattr__(self, attr):
        try:
            result = self.config[attr]
        except KeyError:
            log.warn("%s attribute is not existed, return None" % attr)
            result = None
        return result

    def __setitem__(self, key, value):
        self.config[key] = value

    def __str__(self):
        return str(self.config)

    def pretty_print(self):
        log.info(
            "-----------------------------------------------------------------")
        log.info("config file: %s" % self.config_file)
        for key, value in sorted(
                self.config.items(), key=lambda item: item[0]):
            log.info("%s: %s" % (key, value))
        log.info(
            "-----------------------------------------------------------------")
