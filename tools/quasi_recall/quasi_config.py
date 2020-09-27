#!/usr/bin/env python
# coding=utf-8
"""
 @auth : wangna07@baidu.com
 @date : 2020-08-04 22:10:58
"""

import yaml

class Config:
    """ Global Configuration Management
    """
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
    
    def __getattr__(self, attr):
        return self.config[attr]

config = Config('./config.yaml')
