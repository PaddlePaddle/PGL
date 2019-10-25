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
"""
This file implement a class for model configure.
"""

import datetime
import os
import yaml
import random
import shutil


class Config(object):
    """Implementation of Config class for model configure.

    Args:
        config_file(str): configure filename, which is a yaml file.
        isCreate(bool): if true, create some neccessary directories to save models, log file and other outputs.
        isSave(bool): if true, save config_file in order to record the configure message.
    """

    def __init__(self, config_file, isCreate=False, isSave=False):
        self.config_file = config_file
        self.config = self.get_config_from_yaml(config_file)

        if isCreate:
            self.create_necessary_dirs()

            if isSave:
                self.save_config_file()

    def get_config_from_yaml(self, yaml_file):
        """Get the configure hyperparameters from yaml file.
        """
        try:
            with open(yaml_file, 'r') as f:
                config = yaml.load(f)
        except Exception:
            raise IOError("Error in parsing config file '%s'" % yaml_file)

        return config

    def create_necessary_dirs(self):
        """Create some necessary directories to save some important files.
        """

        time_stamp = datetime.datetime.now().strftime('%m%d_%H%M')
        self.config['trainer']['args']['log_dir'] = ''.join(
            (self.config['trainer']['args']['log_dir'],
             self.config['task_name'], '/'))  # , '.%s/' % (time_stamp)))
        self.config['trainer']['args']['save_dir'] = ''.join(
            (self.config['trainer']['args']['save_dir'],
             self.config['task_name'], '/'))  # , '.%s/' % (time_stamp))) 
        self.config['trainer']['args']['output_dir'] = ''.join(
            (self.config['trainer']['args']['output_dir'],
             self.config['task_name'], '/'))  # , '.%s/' % (time_stamp)))
        #  if os.path.exists(self.config['trainer']['args']['save_dir']):
        #      input('save_dir is existed, do you really want to continue?')

        self.make_dir(self.config['trainer']['args']['log_dir'])
        self.make_dir(self.config['trainer']['args']['save_dir'])
        self.make_dir(self.config['trainer']['args']['output_dir'])

    def save_config_file(self):
        """Save config file so that we can know the config when we look back
        """
        filename = self.config_file.split('/')[-1]
        targetpath = self.config['trainer']['args']['save_dir']
        shutil.copyfile(self.config_file, targetpath + filename)

    def make_dir(self, path):
        """Build directory"""
        if not os.path.exists(path):
            os.makedirs(path)

    def __getitem__(self, key):
        """Return the configure dict"""
        return self.config[key]

    def __call__(self):
        """__call__"""
        return self.config
